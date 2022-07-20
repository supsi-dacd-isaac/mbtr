import numpy as np
import networkx as nx
from functools import lru_cache
from scipy.linalg import eigh
from mbtr.utils import check_pars
import warnings


class Loss:
    """
    Loss function class. A loss is defined by its gradient and Hessians.
    Note that if your specific loss funciton requires some additional argument, you can specify it in the
    required_pars. Upon instantiation, this list will be used to check if loss_kwargs contains all the needed
    parameters. Each class inheriting from :class:`mbtr.losses.Loss` must provide an H_inv method, computing the
    inverse of the Hessian.

    :param lambda_weights: quadratic penalization parameter for the leaves weights
    :param lambda_leaves: quadratic penalization parameter for the number of leaves
    :param loss_kwargs: additional parameters needed for a specific loss type
    """
    required_pars = []  # required pars for which no default is set, which are needed by specific losses

    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        self.lambda_weights = lambda_weights
        self.lambda_leaves = lambda_leaves
        self.kwargs = loss_kwargs
        self.n_dims = None          # this is inferred by Tree or MBT

    def set_dimension(self, n_dims):
        """
        Initialize all the properties which depends on the dimension of the target

        :param n_dims: dimension of the target
        :return: None
        """
        self.n_dims = n_dims

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediciton. This can be loss-specific.

        :param y: target matrix of the training set

        :return: np.ndarray with initial guess
        """
        return np.mean(y, axis=0)

    def get_grad_and_hessian_diags(self, y, y_hat, iteration, leaves_idx):
        """
        Return the loss gradient and loss Hessian's diagonals based on the current model estimation y_hat and target y
        matrices. Instead of returning the full Hessian (a 3rd order tensor), the method returns only the Hessian
        diagonals for each observation, stored in a (n_obs, n_t) matrix. These diagonals are then used by the loss to
        reconstruct the full Hessian with appropriate dimensions and structure. Currently, full Hessians inferred by
        data are not supported.

        :param y: target matrix (n_obs, n_t)
        :param y_hat: current target estimation matrix (n_obs, n_t)
        :param iteration: current number of iteration, generally not needed
        :param leaves_idx: leaves' indexes for each observation in y, (n_obs, 1). This is needed for example by :class:`mbtr.losses.QuadraticQuantileLoss`.

        :return: grad, hessian_diags tuple, each of which is a (n_obs, n_t) matrix
        """
        grad = - (y - y_hat)
        hessian_diags = np.ones_like(grad)
        return grad, hessian_diags

    def eval_optimal_response(self, G, H):
        """
        Evaluate optimal response, given G and H.

        :param G: mean gradient for the current leaf.
        :param H: mean Hessian for the current leaf.

        :return: optimal response under second order loss approximation.
        """
        H_inv = self.H_inv(H)
        optimal_response = H_inv @ G
        return optimal_response

    def eval_optimal_loss(self, G2, H):
        """
        Evaluate the optimal loss (using response obtained minimizing the second order loss approximation).

        :param G2: squared sum of gradients in the current leaf
        :param H: sum of Hessians diags in the current leaf

        :return: optimal loss, scalar
        """
        H_inv = self.H_inv(H)
        optimal_loss = - np.sum(H_inv @ G2)
        return optimal_loss

    def eval(self,  y, y_hat, trees):
        """
        Evaluates the overall loss, which is composed by the tree's loss plus weights and total leaves penalizations

        :param y: observations
        :param y_hat: current :class: `mbtr.MBT` estimations
        :param trees: array of fitted trees up to the current iteartion

        :return: tree loss and regularizations loss tuple, scalars
        """
        w_reg, t_reg, w_smooth = [0, 0, 0]
        for tree in trees:
            # parameter regularization
            w_reg += self.lambda_weights * \
                     np.sum([r ** 2 for r in nx.get_node_attributes(tree.g, 'response').values()])
            # leaves regularization
            t_reg += self.lambda_leaves * len(nx.get_node_attributes(tree.g, 'response').values())

        # squared error
        se = self.tree_loss(y, y_hat)
        regularization = t_reg + w_reg

        return se, regularization

    @lru_cache(maxsize=1024)
    def H_inv(self, H):
        """
        Computes the inverse of the Hessian, given the Hessian's diagonal of the current leave. The default implements
        MSE inverse.

        :param H: current leaf Hessian's diagonal (n_t)

        :return: inv(H), (n_t, n_t)
        """
        return np.eye(self.n_dims) / (np.mean(H) + self.lambda_weights)

    def tree_loss(self, y, y_hat):
        """
        Compute the tree loss (without penalizations)

        :param y: observations of the target on the traning set
        :param y_hat: current estimation of the MBT

        :return: tree loss
        """
        return np.sum((y - y_hat) ** 2)


class MSE(Loss):
    """
    Mean Squared Error loss, a.k.a. L2, Ordinary Least Squares.

    .. math::

            \\mathcal{L} = \\Vert y - w\\Vert_2^2 + \\frac{1}{2} w^T \\Lambda w

    where :math:`\\Lambda` is the quadratic punishment matrix.

    :param lambda_weights:  quadratic penalization parameter for the leaves weights
    :param lambda_leaves: quadratic penalization parameter for the number of leaves
    :param loss_kwargs: additional parameters needed for a specific loss type
    """
    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)

    def set_dimension(self, n_dim):
        """
        Initialize all the properties which depends on the dimension of the target

        :param n_dims: dimension of the target
        :return: None
        """
        super().set_dimension(n_dim)

    def eval_optimal_loss(self, G2, H):
        """
        Evaluate the optimal loss (using response obtained minimizing the second order loss approximation).

        :param G2: squared sum of gradients in the current leaf
        :param H: sum of Hessians diags in the current leaf

        :return: optimal loss, scalar
        """

        H_inv = self.H_inv(np.mean(H))
        optimal_loss = - np.sum(H_inv @ G2)
        return optimal_loss

    def eval_optimal_response(self, G, H):
        """
        Evaluate optimal response, given G and H.

        :param G: mean gradient for the current leaf.
        :param H: mean Hessian for the current leaf.

        :return: optimal response under second order loss approximation.
        """
        H_inv = self.H_inv(np.mean(H))
        optimal_response = H_inv @ G
        return optimal_response


class TimeSmoother(Loss):
    """
    Time-smoothing loss function. Penalizes the time-derivative of the predicted signal.

    .. math::

            \\mathcal{L} = \\frac{1}{2}\\Vert y-w\\Vert_2^{2} + \\frac{1}{2} w^T \\left(\\lambda_s D^T D
            + \\lambda I \\right) w

    where :math:`D` is the second order difference matrix

    .. math::

            D=\\left[\\begin{array}{rrrrrr}
            1 & -2 & 1 & & &

            & 1 & -2 & 1 & &

            & & \\ddots & \\ddots & \\ddots &

            & & & 1 & -2 & 1

            & & & & 1 & -2 & 1
            \\end{array}\\right]

    and :math:`\\lambda_s` is the coefficient for the quadratic penalization of time-derivatives.
    Required parameters: lambda_smooth: coefficient for the quadratic penalization of time-derivatives
    """
    required_pars = ['lambda_smooth']

    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        check_pars(self.required_pars, **loss_kwargs)
        self.lambda_smooth = self.kwargs['lambda_smooth']
        self.lambdas = None
        self.Q_inv = None
        self.D = None

    def set_dimension(self, n_dim):
        """
        Initialize all the properties which depends on the dimension of the target

        :param n_dims: dimension of the target

        :return: None
        """
        super().set_dimension(n_dim)
        self.D = self.build_filter_mat(self.n_dims)
        self.lambdas, self.Q = eigh(np.eye(self.n_dims) * self.lambda_weights + self.lambda_smooth * self.D.T @ self.D)
        self.Q_inv = np.linalg.inv(self.Q)
        self.H_inv = lambda H: self.compute_fast_H_inv(np.mean(H))

    @staticmethod
    def build_filter_mat(n):
        """
        Build the second order difference matrix

        :param n: target dimension

        :return: D, second order difference matrix
        """
        D = (np.diag(np.ones(n)) + np.diag(-np.ones(n - 1) * 2, 1) + np.diag(np.ones(n - 2), 2))[:n - 2, :]
        return D

    def update_smoothing_mat(self, smoothing_weights):
        self.D = self.build_filter_mat(self.n_dims)
        self.D = self.D * np.reshape(smoothing_weights, (-1, 1))
        self.H_inv = lambda n: np.linalg.inv(np.eye(self.n_dims) * (n + self.lambda_weights)
                                             + self.lambda_smooth * self.D.T @ self.D)

    @lru_cache(maxsize=1024)
    def compute_fast_H_inv(self,n):
        return self.Q @ np.diag(1/(self.lambdas + n)) @ self.Q_inv


class LatentVariable(Loss):
    """
    Loss for the hierarchical reconciliation problem, in the form:

    .. math::

        \\mathcal{L} = \\Vert y - S x\\Vert_2^2

    where :math:`S` is the hierarchy matrix. The initial guess is the mean of the last columns of y.
    """

    required_pars = ['S', 'precision']

    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        check_pars(self.required_pars, **loss_kwargs)
        self.S = self.kwargs['S']
        self.precision = self.kwargs['precision']
        self.S_inv = np.linalg.inv(self.S.T@self.S)
        self.lambdas, self.Q = eigh(self.S.T @ self.kwargs['precision'] @ self.S)
        self.Q_inv = np.linalg.inv(self.Q)
        self.H_inv = lambda H: self.compute_H_inv(np.mean(H))
        self.n_dims = self.S.shape[1]

    def set_dimension(self, n_dims):
        """
        For the latent loss the number of dimensions is equal to the second dimension of the S matrix, and must not
        be inferred from the target
        """
        pass

    def get_initial_guess(self, y):
        """
        The initial guess is generated from the last columns of the target matrix, as:

        .. math::

                y_0 = \\left( \\mathbb{E} y_b \\right) S^T

        where :math:`\\mathbb{E}` is the expectation (row-mean), :math:`S` is the hierarchy matrix, and :math:`y_b`
        stands for the last columns of y, with dimension (n_obs, n_b), where n_b is the number of bottom series.

        :param y: target matrix of the training set

        :return: np.ndarray with initial guess
        """
        start = y.shape[1] - self.n_dims
        stop = start + self.n_dims
        assert start > 0, 'dimension of target ({}) must be higher than the searched latent ' \
                          'variables ({})'.format(y.shape[1], self.n_dims)
        y_0 = np.mean(y[:, start:stop] @ self.S.T, axis=0)
        return y_0

    def get_grad_and_hessian_diags(self,y, y_hat, iteration, leaves_idx):
        grad = - (y - y_hat) @ self.precision
        hessian_diags = np.ones_like(grad)
        return grad, hessian_diags

    @lru_cache(maxsize=1024)
    def compute_fast_H_hat(self, n):
        return self.Q @ np.diag(1/(self.lambdas + self.lambda_weights/(n+1e-12))) @ self.Q_inv / (n+1e-12)

    @lru_cache(maxsize=1024)
    def compute_H_inv(self,n):
        return np.eye(self.S.shape[0]) / (n + self.lambda_weights)

    def eval_optimal_response(self, G, H):
        """
        Evaluate optimal response, given G and H.

        :param G: mean gradient for the current leaf.
        :param H: mean Hessian for the current leaf.

        :return: optimal response under second order loss approximation.
        """
        optimal_response = (self.compute_fast_H_hat(np.mean(H))@ (G @ self.S)) @ self.S.T
        return optimal_response

    def eval_optimal_loss(self, yy, H):
        """
        Evaluate the optimal loss (using response obtained minimizing the second order loss approximation).

        :param G2: squared sum of gradients in the current leaf
        :param H: sum of Hessians diags in the current leaf

        :return: optimal loss, scalar
        """

        H_inv = self.H_inv(H)
        optimal_loss = -np.sum((H_inv @ yy @ self.precision))
        return optimal_loss


class QuantileLoss(Loss):

    required_pars = ['alphas']

    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        check_pars(self.required_pars, **loss_kwargs)
        self.alphas = self.kwargs['alphas']
        self.tree_loss = self.quantile_loss

    def get_initial_guess(self, y):
        """
        The initial guess are the alpha quantiles of the target matrix y.

        :param y: target matrix of the training set

        :return: np.ndarray with initial guess
        """
        alphas = self.alphas
        return np.quantile(y, alphas).reshape(1, -1)

    def H_inv(self, H):
        return np.diag(1 / (H + self.lambda_weights))

    def quantile_loss(self, y, q_hat):
        """
        Quantile loss function, a.k.a. pinball loss.

        .. math::

            \\epsilon (y,\\hat{q})_{\\alpha} &= \\hat{q}_{\\alpha} - y

            \\mathcal{L} (y,\\hat{q})_{\\alpha} &= \\epsilon (y,\\hat{q})_{\\alpha} \\left( I_{\\epsilon_{\\alpha}\\geq 0} -\\alpha \\right)

        :param y: observations of the target on the traning set
        :param q_hat: current estimation matrix of the quantiles

        :return: tree loss
        """
        loss = np.zeros(len(self.alphas))
        for i, alpha in enumerate(self.alphas):
            I = q_hat[:, [i]] < y
            loss[i] = np.sum((alpha * I + (1-alpha)*(~I)) * np.abs(y-q_hat[:, [i]]))
        return np.sum(loss)

    def get_grad_and_hessian_diags(self, y, y_hat, iteration, leaves_idx):
        loss_grad, hessian = [[],[]]
        for i,alpha in enumerate(self.alphas):
            err = y - y_hat[:, [i]]
            shift = np.log((1 - alpha) / alpha)
            loss_grad_i = -(np.exp(err+shift)/(1 + np.exp(err+shift)) + alpha-1)
            hessian_i = err+shift
            hessian_i[hessian_i <= 10] = np.exp(
                hessian_i[hessian_i <= 10])/((1 + np.exp(hessian_i[hessian_i <= 10]))**2) + 1e-3
            hessian_i[hessian_i>10] = 0
            loss_grad.append(loss_grad_i)
            hessian.append(hessian_i)

        loss_grad = np.hstack(loss_grad)
        hessian = np.hstack(hessian)
        return loss_grad, hessian

    def exact_response(self, y):
        response = np.zeros(len(self.alphas))
        for i in range(len(response)):
            response[i] = np.quantile(y[:, i], self.alphas[i])
        return response


class QuadraticQuantileLoss(Loss):

    required_pars = ['alphas']

    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        check_pars(self.required_pars, **loss_kwargs)
        self.alphas = self.kwargs['alphas']

    def get_initial_guess(self, y):
        """
        The initial guess are the alpha quantiles of the target matrix y.

        :param y: target matrix of the training set

        :return: np.ndarray with initial guess
        """
        alphas = self.alphas
        return np.quantile(y, alphas).reshape(1, -1)

    def H_inv(self, H):
        return np.diag(1 / (H + self.lambda_weights))

    def tree_loss(self, y, y_hat):
        """
        Compute the tree loss (without penalizations)

        :param y: observations of the target on the traning set
        :param y_hat: current estimation of the MBT

        :return: tree loss
        """
        loss = np.zeros(len(self.alphas))
        for i, alpha in enumerate(self.alphas):
            I = y_hat[:, [i]] < y
            loss[i] = np.sum((alpha*I + (1 - alpha) * (~I)) * np.abs(y - y_hat[:, [i]]))
        return np.sum(loss)

    def get_grad_and_hessian_diags(self,y, y_hat, iteration, leaves_idx):
        err = y - y_hat
        grad, hessian_diags = [[],[]]
        k = np.ones(len(self.alphas)) * 1e5
        for i, alpha in enumerate(self.alphas):
            grad_alpha = np.zeros((len(err),1))
            hessian_alpha = np.zeros((len(err), 1))
            for leaf_idx in leaves_idx:
                err_leaf = err[leaf_idx,i]
                lefts_idx, rights_idx = [err_leaf<=0, err_leaf>0]
                norm_l = np.sum(err_leaf[lefts_idx]) if np.any(lefts_idx) else 1
                norm_r = np.sum(err_leaf[rights_idx]) if np.any(rights_idx) else 1
                # next two lines were due to an error in the formula
                #grad_alpha[leaf_idx] = (err_leaf * (-2*k[i] + lefts_idx *(1-alpha-k[i]/norm_l) + rights_idx *(alpha+k[i]/norm_r))).reshape(-1,1)
                #hessian_alpha[leaf_idx] = -(-2*k[i] + lefts_idx *(1-alpha-k[i]/norm_l) + rights_idx *(alpha+k[i]/norm_r) ).reshape(-1,1)
                grad_alpha[leaf_idx] = -((1 - alpha + k[i] * err_leaf / norm_r) * rights_idx
                                         + (-alpha + k[i] * err_leaf / norm_l) * lefts_idx - 2 * k[i] / n).reshape(-1,
                                                                                                                   1)
                hessian_alpha[leaf_idx] = -(k[i] * rights_idx / norm_r + k[i] * lefts_idx / norm_l).reshape(-1, 1)
            grad.append(grad_alpha)
            hessian_diags.append(hessian_alpha)
        grad = np.hstack(grad)
        hessian_diags = np.hstack(hessian_diags)
        return grad, hessian_diags

    def exact_response(self, y):
        response = np.zeros(len(self.alphas))
        for i in range(len(response)):
            response[i] = np.quantile(y[:,i], self.alphas[i])
        return response


class FourierLoss(Loss):
    """
    Loss for the Fourier regression:

    .. math::

        \\mathcal{L} = \\Vert y - P x\\Vert_2^2

    where :math:`P` is the projection matrix:

    .. math::

        P=\\left[\\left[\\cos \\left(k \\frac{2 \\pi t}{n_{t}}\\right)^{T}, \\sin \\left(k
        \\frac{2 \\pi t}{n_{t}}\\right)^{T}\\right]^{T}\\right]_{k \\in \\mathcal{K}}

    """
    required_pars = ['n_harmonics']

    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)
        check_pars(self.required_pars, **loss_kwargs)
        self.n_harmonics = self.kwargs['n_harmonics']
        self.cosines = None
        self.sines = None
        self.P = None

    def set_dimension(self, n_dim):
        """
        Initialize all the properties which depends on the dimension of the target

        :param n_dims: dimension of the target
        :return: None
        """
        super().set_dimension(n_dim)
        self.cosines = np.cos(np.outer(np.arange(self.n_harmonics), np.arange(self.n_dims) * 2 * np.pi / self.n_dims))
        self.sines = np.sin(np.outer(np.arange(self.n_harmonics), np.arange(self.n_dims) * 2 * np.pi / self.n_dims))
        self.P = np.vstack([self.sines, self.cosines])

    def get_initial_guess(self, y):
        """
        Return an initial guess for the prediciton. This can be loss-specific.

        :param y: target matrix of the training set

        :return: np.ndarray with initial guess
        """
        return np.tile(np.mean(y), y.shape[1])

    @lru_cache(maxsize=1024)
    def projection_matrix(self, n):
        """
        Return projection matrix for the Fourier coefficient estimation.

        :param n: number of observations

        :return: projection matrix P, (2*n_harmonics, n_t) where n_harmonics is the number of harmonics to fit, n_t the
        target dimension
        """
        return self.P/(self.n_dims * n / 2)

    def eval_optimal_response(self, G, H):
        """
        Evaluate optimal response, given G and H.

        :param G: mean gradient for the current leaf.
        :param H: mean Hessian for the current leaf.

        :return: optimal response under second order loss approximation.
        """
        P = self.projection_matrix(np.mean(H))
        return np.sum(self.P * (P @ G).reshape(-1,1), axis=0)

    def eval_optimal_loss(self, G2, H):
        """
        Evaluate the optimal loss (using response obtained minimizing the second order loss approximation).

        :param G2: squared sum of gradients in the current leaf
        :param H: sum of Hessians diags in the current leaf

        :return: optimal loss, scalar
        """
        optimal_loss = - np.sum(G2 / np.mean(H) / self.n_dims)

        return optimal_loss


class LinRegLoss(Loss):
    def __init__(self, lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        super().__init__(lambda_weights, lambda_leaves, **loss_kwargs)

    def set_dimension(self, n_dim):
        """
        Initialize all the properties which depends on the dimension of the target

        :param n_dims: dimension of the target
        :return: None
        """
        super().set_dimension(n_dim)

    def eval_optimal_response(self, G, x):
        """
        Evaluate optimal response, given G and x. This is done computing a Ridge regression with intercept

        .. math::

                w = \\left(\\tilde{x}^T \\tilde{x} + \\lambda I \\right)^{-1} \\left(\\tilde{x}^T G \\right)

        where :math:`\\tilde{x}` is the :math:`x` matrix augmented with an unitary column and :math:`\\lambda` is the
        Ridge coefficient.

        :param G: gradient for the current leaf.
        :param x: linear regression features for the current leaf.

        :return: optimal response under second order loss approximation.
        """
        x = np.hstack([x,np.ones((x.shape[0],1))])
        P = np.eye(x.shape[1]) * self.lambda_weights
        coeffs = np.linalg.pinv(x.T@x + P) @ (x.T@G)
        if np.any(coeffs)>1e3:
            warnings.warn('Linear regression coeffs seem too high: max is {}!'.format(np.max(coeffs)))
        return coeffs

    def eval_optimal_loss(self, G, x):
        """
        Evaluate the optimal loss (using response obtained minimizing the second order loss approximation).

        :param G: gradient for the current leaf.
        :param x: linear regression features for the current leaf.

        :return: optimal loss, scalar
        """
        coeffs = self.eval_optimal_response(G,x)
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        y_hat = x@coeffs
        loss = np.sum((G-y_hat)**2)
        return loss