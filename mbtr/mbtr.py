import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mbtr.losses import LinRegLoss, QuantileLoss, QuadraticQuantileLoss
# package required for plotting the tree; installation issues under windows
# from networkx.drawing.nx_agraph import graphviz_layout
from time import time
from numba import jit, float32
from mbtr import LOSS_MAP
from tqdm import tqdm
from mbtr.utils import set_figure
from copy import deepcopy


class Tree:
    """
    Tree class. Fits both univarite and multivariate targets. It implements histogram search for the decision of the
    splitting points.

    :param n_q: number of quantiles for the split search
    :param min_leaf: minimum number of observations in one leaf. This parameter greatly affect generalization abilities.
    :param loss_type: loss type for choosing the best splits. Currently the following losses are implemented:

                      mse: mean squared error loss, a.k.a. L2, ordinary least squares

                      time_smoother: mse with an additional penalization on the second order differences of the response
                      function. Requires to pass also lambda_smooth parameter.

                      latent_variable: generate response function of dimension n_t from an arbitrary linear combination
                      of n_r responses. This requires to pass also S and precision pars.

                      linear_regression: mse with linear response function. Using this loss function, when calling fit
                      and predict methods, one must also pass x_lr as additional argument, which is the matrix of
                      features used to train the linear response inside the leaf (which can be different from the
                      features used to grow the tree, x).

                      fourier: mse with linear response function, fitted on the first n_harmonics (where the
                      fundamental has wave-lenght equal to the target output). This requires to pass also the
                      n_harmonics parameter.

                      quantile: quantile loss function, a.k.a. pinball loss. This requires to pass also the alphas
                      parameter, a list of quantiles to be fitted.

                      quadratic_quantile: quadratic quantile loss function tailored for trees. It has a non-discontinuos
                      derivative. This requires to pass also the alphas parameter, a list of quantiles to be fitted.
    :param lambda_weights: coefficient for the quadratic regularization of the response's parameters
    :param lambda_leaves: coefficient for the quadratic regularization of the total number of leaves. This is only used
    when the Tree is used as a weak learner by MBT.
    :param loss_kwargs: possible additional arguments for the loss function

    """
    def __init__(self, n_q: int = 10, min_leaf: int = 100, loss_type: str = "mse",
                 lambda_weights: float = 0.1, lambda_leaves: float = 0.1, **loss_kwargs):
        self.n_q = n_q
        self.min_leaf = min_leaf
        self.loss_pars = {**{"loss_type": loss_type, "lambda_weights": lambda_weights, "lambda_leaves": lambda_leaves,
                             **loss_kwargs}}
        self.g = nx.DiGraph()
        self.g.add_node('origin', leaf=False, loss=np.inf)
        self.loss = LOSS_MAP[self.loss_pars['loss_type']](**self.loss_pars)

    def fit(self, x, y, hessian=None, learning_rate=1.0, x_lr=None):
        """
        Fits a tree using the features specified in the matrix :math:`x\\in\\mathbb{R}^{n_{obs} \\times n_{f}}`, in order
        to predict the targets in the matrix :math:`y\\in\\mathbb{R}^{n_{obs} \\times n_{t}}`, where :math:`n_{obs}` is
        the number of observations, :math:`n_{f}` the number of features and :math:`n_{t}` the dimension of the target.

        :param x: feature matrix, np.ndarray.
        :param y: target matrix, np.ndarray.
        :param hessian: diagonals of the hessians :math:`\\in\\mathbb{R}^{n_{obs} \\times n_{t}}`. If None, each entry
                        is set equal to one (this will result in the default behaviour under MSE loss). Default: None
        :param learning_rate: learning rate used by the MBT instance. Default: 1
        :param x_lr: features for fitting the linear response inside the leaves. This is only required if a LinearLoss
                      is being used.

        """
        # set loss dimensionality
        self.loss.set_dimension(y.shape[1])

        if hessian is None:
            hessian = np.ones_like(y)
        if type(self.loss) == LinRegLoss:
            self._fit_node_lr(x, y, x_lr, 'origin', learning_rate=learning_rate, hessian=hessian)
        else:
            self._fit_node(x, y, 'origin', learning_rate=learning_rate, hessian=hessian)

    def predict(self, x, x_lr=None):
        """
        Predicts the target based on the feature matrix x (and linear regression features x_lr).

        param: x: feature matrix, np.ndarray.
        param: x_lr: linear regression feature matrix, np.ndarray.

        :return: target's predictions
        """
        r = []
        if x_lr is None:
            for x_i in x:
                r.append(self._predict_node(x_i, 'origin'))
        else:
            for x_i, x_lr_i in zip(x,x_lr):
                r.append(self._predict_node(x_i, 'origin', x_lr_i))
        return np.vstack(r)

    def _predict_node(self, x:np.ndarray, node, x_lr:np.ndarray=None):
        """
        Recursively explores the tree and retrieve the response

        :param x: matrix of covariates, (n_obs, n_vars)
        :param node: current node of the tree
        :param x_lr: matrix of linear regression covariates. Only required if we are fitting linear regressions inside
         the nodes

        :return: response: matrix of responses
        """
        if self.g.nodes[node]['leaf']:
            if type(self.loss) == LinRegLoss:
                response = np.hstack([x_lr,1]) @ self.g.nodes[node]['response']
            else:
                response = self.g.nodes[node]['response']
        else:
            v = self.g.nodes[node]['variable']
            l = self.g.nodes[node]['level']
            if x[v] < l:
                node = [i for i in self.g.successors(node) if self.g.nodes[i]['side'] == 'l']
            else:
                node = [i for i in self.g.successors(node) if self.g.nodes[i]['side'] == 'r']
            response = self._predict_node(x, node[0], x_lr)
        return response

    def _fit_node(self, x:np.ndarray, y:np.ndarray, node, learning_rate=1.0, hessian=None):
        """
        Recursively fits the tree using second order approximation (gradient and hessian) of the objective function
        :param x: matrix of covariates, (n_obs, n_vars)
        :param y: matrix of target, (n_obs, n_t)
        :param node: current node of the tree
        :param: learning_rate: discount factor. Undershoot the weak learner response to dampen the learning process
        """
        if x.shape[0] < self.min_leaf:
            self.g.nodes[node]['leaf'] = True
            self.g.nodes[node]['response'] = self.loss.eval_optimal_response(
                np.sum(y, axis=0), np.sum(hessian, axis=0)) * learning_rate
            return
        else:
            is_leaf = True
            self.g.nodes[node]['loss'] = self.loss.eval_optimal_loss(np.sum(y,axis=0) ** 2, np.sum(hessian, axis=0))
            current_loss = np.copy(self.g.nodes[node]['loss'])

            # for all the variables, find best split point
            for i in range(x.shape[1]):
                # find quantiles
                qs_i = np.unique(np.quantile(x[:,i],np.linspace(1/self.n_q,1-1/self.n_q,self.n_q)))
                G2_left, G2_right = leaf_stats(y, np.hstack([-np.inf,qs_i,np.inf]), x[:, i], order=1)
                H_left, H_right = leaf_stats(hessian, np.hstack([-np.inf, qs_i, np.inf]), x[:, i], order=2)

                # for all the quantiles of the current variables, find best quantile split
                for j, q_i in enumerate(qs_i):
                    loss = self.compute_loss(G2_left, G2_right, H_left, H_right, j)
                    if loss < current_loss:
                        is_leaf = False
                        current_loss = np.copy(loss)
                        self.g.nodes[node]['variable'] = i
                        self.g.nodes[node]['level'] = q_i

            # if a split reducing the total loss has been found, split the node at best variable/level
            self.g.nodes[node]['leaf'] = is_leaf
            if is_leaf:
                self.g.nodes[node]['response'] = self.loss.eval_optimal_response(
                    np.sum(y, axis=0), np.sum(hessian, axis=0)) * learning_rate
            else:
                node_name = self._add_new_node(node)
                filt_1 = x[:, self.g.nodes[node]['variable']] < self.g.nodes[node]['level']
                filt_2 = x[:, self.g.nodes[node]['variable']] >= self.g.nodes[node]['level']
                if (np.sum(filt_1) == 0) or (np.sum(filt_2) == 0):
                    self.g.nodes[node]['response'] = self.loss.eval_optimal_response(
                        np.sum(y, axis=0), np.sum(hessian, axis=0)) * learning_rate
                    self.g.nodes[node]['leaf'] = True
                    return
                else:
                    self._fit_node(x[filt_1, :], y[filt_1, :], node_name, learning_rate, hessian[filt_1, :])
                    self._fit_node(x[filt_2, :], y[filt_2, :], node_name+1, learning_rate, hessian[filt_2, :])

    def _fit_node_lr(self, x, y, x_lr, node, learning_rate, hessian):
        if x.shape[0] < self.min_leaf:
            self.g.nodes[node]['leaf'] = True
            self.g.nodes[node]['response'] = self.loss.eval_optimal_response(
                y, x_lr) * learning_rate
            return
        else:
            is_leaf = True
            self.g.nodes[node]['loss'] = self.loss.eval_optimal_loss(y, x_lr)
            current_loss = np.copy(self.g.nodes[node]['loss'])

            # for all the variables, find best split point
            for i in range(x.shape[1]):
                # find quantiles
                qs_i = np.unique(np.quantile(x[:, i], np.linspace(1/self.n_q, 1-1/self.n_q, self.n_q)))
                # for all the quantiles of the current variables, find best quantile split
                for j, q_i in enumerate(qs_i):
                    filt = x[:, i] <= q_i
                    loss_l = self.loss.eval_optimal_loss(y[filt,:], x_lr[filt,:])
                    loss_r = self.loss.eval_optimal_loss(y[~filt,:], x_lr[~filt,:])
                    loss = loss_l + loss_r
                    if loss < current_loss:
                        is_leaf = False
                        current_loss = np.copy(loss)
                        self.g.nodes[node]['variable'] = i
                        self.g.nodes[node]['level'] = q_i

            # if a split reducing the total loss has been found, split the node at best variable/level
            self.g.nodes[node]['leaf'] = is_leaf
            if is_leaf:
                self.g.nodes[node]['response'] = self.loss.eval_optimal_response(
                    y, x_lr) * learning_rate
            else:
                node_name = self._add_new_node(node)
                filt_1 = x[:, self.g.nodes[node]['variable']] < self.g.nodes[node]['level']
                filt_2 = x[:, self.g.nodes[node]['variable']] >= self.g.nodes[node]['level']
                if (np.sum(filt_1) == 0) or (np.sum(filt_2) == 0):
                    self.g.nodes[node]['response'] = self.loss.eval_optimal_response(
                        y, x_lr) * learning_rate
                    self.g.nodes[node]['leaf'] = True
                    return
                else:
                    self._fit_node_lr(x[filt_1, :], y[filt_1, :], x_lr[filt_1,:],
                                      node_name, learning_rate,hessian[filt_1,:])
                    self._fit_node_lr(x[filt_2, :], y[filt_2, :], x_lr[filt_2,:],
                                      node_name + 1, learning_rate,hessian[filt_2,:])

    # this funciton requires graphviz, which has installation issues under windows
    '''
    def plot_tree(self):
        colors = [self.g.nodes[n]['loss'] for n in self.g.nodes()]
        pos = graphviz_layout(self.g,prog='dot')
        nc = nx.draw_networkx_nodes(self.g, pos, node_size=30, alpha=0.8, node_color=colors, cmap=plt.cm.plasma)
        nx.draw_networkx_edges(self.g, pos, alpha=0.2)
        nx.draw_networkx_labels(self.g,pos, font_size=10)
        plt.colorbar(nc,label='attribute')
    '''

    def _find_regions(self, x, y):
        leaves = nx.get_node_attributes(self.g, 'response').keys()
        filters = []
        for leaf in leaves:
            path = nx.shortest_path(self.g, 'origin', leaf)
            node_filter = np.ones(x.shape[0], dtype=bool)
            for i in range(len(path)-1):
                v_p = self.g.nodes[path[i]]['variable']
                level_p = self.g.nodes[path[i]]['level']
                side = self.g.nodes[path[i+1]]['side']
                if side == 'l':
                    node_filter_i = x[:,v_p] < level_p
                else:
                    node_filter_i = x[:, v_p] >= level_p
                node_filter &= node_filter_i
            self.g.nodes[leaf]['y'] = y[node_filter,:]
            self.g.nodes[leaf]['filter'] = node_filter
            filters.append(node_filter)
            del node_filter
        return filters

    def _refit(self, x, y, discount=1.0, x_lr=None):
        if "exact_response" not in dir(self.loss):
            return
        self._find_regions(x, y)
        leaves = nx.get_node_attributes(self.g, 'response').keys()
        for l in leaves:
            y_l = self.g.nodes[l]['y']
            filter_l = self.g.nodes[l]['filter']
            if x_lr is None:
                self.g.nodes[l]['response'] = self.loss.exact_response(y_l) * discount
            else:
                x_lr_l = x_lr[filter_l,:]
                self.g.nodes[l]['response'] = self.loss.exact_response(y_l, x_lr_l) * discount
            self.g.nodes[l]['y'] = None

    def _add_new_node(self, node):
        node_name = self.g.number_of_nodes() + 1
        self.g.add_node(node_name, loss=np.inf, side='l')
        self.g.add_node(node_name + 1, loss=np.inf, side='r')
        self.g.add_edges_from(((node, node_name), (node, node_name + 1)))
        return node_name

    @staticmethod
    def _compute_bin_sums(y, edges, x):
        accumarray_idx = np.digitize(x,edges)
        partial_sums = np.zeros((len(edges)+1, y.shape[1]))
        for i in range(y.shape[1]):
            partial_sums[:,i] = np.bincount(accumarray_idx, weights=y[:,i]).ravel()

        accumulated_bin_sums = np.cumsum(partial_sums, axis=0)
        accumulated_elements = np.cumsum(np.bincount(accumarray_idx), axis=0)
        yy_left = accumulated_bin_sums ** 2
        yy_right = (accumulated_bin_sums[-1,:]-accumulated_bin_sums) ** 2
        return yy_left, yy_right, accumulated_elements

    def compute_loss(self, G2_left, G2_right, H_left, H_right, j):
        loss_l = self.loss.eval_optimal_loss(G2_left[j], H_left[j])
        loss_r = self.loss.eval_optimal_loss(G2_right[j], H_right[j])
        loss = loss_l + loss_r
        return loss


@jit('float64[:,:](float64[:,:], float64[:], float64[:])',
     nopython=True, parallel=False)
def bin_sums(y, edges, x):
    edges = edges
    y_ij = np.zeros((len(edges)-1,y.shape[1]))
    for i in np.arange(len(edges)-1):
        index = ((edges[i] <= x) * (x < edges[i+1]))
        y_ij[i,:] = np.sum(y[index,:],axis=0) + y_ij[i-1,:]
    return y_ij


@jit('Tuple([float64[:,:],float64[:,:]])(float64[:,:], float64[:], float64[:], int16)',
     nopython=True, parallel=False)
def leaf_stats(y, edges, x, order):
    y_ij = bin_sums(y, edges, x)
    if order == 1:
        s_left = y_ij ** 2
        s_right = (y_ij[-1,:]-y_ij) ** 2
    elif order == 2:
        s_left = y_ij
        s_right = (y_ij[-1, :] - y_ij)
    else:
        s_left, s_right = None, None
    return s_left, s_right


class MBT:
    """
    Multivariate Boosted Tree class. Fits a multivariate tree using boosting.

    :param n_boosts: maximum number of boosting rounds. Default: 20
    :param early_stopping_rounds: if the total loss is non-decreasing after early_stopping_rounds, stop training.
                                  The final model is the one which achieved the lowest loss up to the final iteration.
                                  Default: 3.
    :param learning_rate: in [0, 1]. A learning rate < 1 helps reducing overfitting. Default: 0.1.
    :param val_ratio: in [0,1]. If provided, the early stop is triggered by the loss computed on a validation set,
                      randomly extracted from the training set. The length of the validation set is
                      val_ratio * len (training set). Default: 0.
    :param n_q: number of quantiles for the split search. Default: 10.
    :param min_leaf: minimum number of observations in one leaf. This parameter greatly affect generalization abilities.
                     Default: 100.
    :param loss_type: loss type for choosing the best splits. Currently the following losses are implemented:

                      mse: mean squared error loss, a.k.a. L2, ordinary least squares

                      time_smoother: mse with an additional penalization on the second order differences of the response
                      function. Requires to pass also lambda_smooth parameter.

                      latent_variable: generate response function of dimension n_t from an arbitrary linear combination
                      of n_r responses. This requires to pass also S and precision pars.

                      linear_regression: mse with linear response function. Using this loss function, when calling fit
                      and predict methods, one must also pass x_lr as additional argument, which is the matrix of
                      features used to train the linear response inside the leaf (which can be different from the
                      features used to grow the tree, x).

                      fourier: mse with linear response function, fitted on the first n_harmonics (where the
                      fundamental has wave-lenght equal to the target output). This requires to pass also the
                      n_harmonics parameter.

                      quantile: quantile loss function, a.k.a. pinball loss. This requires to pass also the alphas
                      parameter, a list of quantiles to be fitted.

                      quadratic_quantile: quadratic quantile loss function tailored for trees. It has a non-discontinuos
                      derivative. This requires to pass also the alphas parameter, a list of quantiles to be fitted.
    :param lambda_weights: coefficient for the quadratic regularization of the response's parameters. Default: 0.1
    :param lambda_leaves: coefficient for the quadratic regularization of the total number of leaves. This is only used
                          when the Tree is used as a weak learner by MBT. Default: 0.1
    :param verbose: in {0,1}. If set to 1, the MBT return fitting information at each iteration.
    :param refit: if True, if the loss function has an "exact_response" method, use it to refit the tree
    :param loss_kwargs: possible additional arguments for the loss function
    """
    def __init__(self, n_boosts: int = 20, early_stopping_rounds: int = 3, learning_rate: float = 0.1,
                 val_ratio: int = 0, n_q: int = 10, min_leaf: int = 100, loss_type: str = "mse",
                 lambda_weights: float = 0.1, lambda_leaves: float = 0.1, verbose: int = 0, refit=True, **loss_kwargs):
        self.n_boosts = n_boosts
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate
        self.val_ratio = val_ratio
        self.refit = refit
        self.tree_pars = {**{'n_q': n_q, 'min_leaf': min_leaf, 'loss_type': loss_type, 'lambda_weights': lambda_weights,
                             'lambda_leaves': lambda_leaves}, **loss_kwargs}

        self.trees = []
        self.y_0 = None
        self.verbose = verbose

    def fit(self, x, y, do_plot=False, x_lr=None):
        """
         Fits an MBT using the features specified in the matrix :math:`x\\in\\mathbb{R}^{n_{obs} \\times n_{f}}`, in
         order to predict the targets in the matrix :math:`y\\in\\mathbb{R}^{n_{obs} \\times n_{t}}`,
         where :math:`n_{obs}` is the number of observations, :math:`n_{f}` the number of features and :math:`n_{t}`
         the dimension of the target.

        :param x: feature matrix, np.ndarray.
        :param y: target matrix, np.ndarray.
        :param x_lr: features for fitting the linear response inside the leaves. This is only required if a LinearLoss
                      is being used.
        """
        # divide in training and validation sets (has effect only if pars['val_ratio'] was set
        x_tr, x_val, y_tr, y_val, x_lr_tr, x_lr_val = self._validation_split(x, y, x_lr)
        lowest_loss = np.inf
        best_iter = 0
        self.trees = []
        t_init = time()
        if do_plot:
            fig, ax = set_figure((5, 4))

        for iter in tqdm(range(self.n_boosts)):
            t0 = time()
            tree = Tree(**self.tree_pars)
            if iter == 0:
                y_hat = self._fit_initial_guess(tree, x_tr, y_tr)
                y_hat_val = np.copy(y_hat)
                neg_grad, hessian = self._get_neg_grad_and_hessian_diags(tree, y_tr, y_hat, 0, x)

            # fit the tree
            tree.fit(x_tr, neg_grad, learning_rate=self.learning_rate, hessian=hessian, x_lr=x_lr_tr)

            # if loss function has an "exact_response" method, use it to refit the tree
            if self.refit:
                tree._refit(x_tr, y_tr - y_hat, self.learning_rate, x_lr=x_lr_tr)

            self.trees.append(tree)
            y_hat = y_hat + tree.predict(x_tr,x_lr_tr)
            y_hat_val = y_hat_val + tree.predict(x_val, x_lr_val)
            se, regularization = tree.loss.eval(y_val, y_hat_val, self.trees)
            terminal_leaves = len(nx.get_node_attributes(tree.g, 'response').values())
            if self.verbose == 1:
                tqdm.write('   Iteration: {} fitted in {:0.1e} sec, total loss: {:0.3e}, squared err: {:0.2e},  '
                      'regularization: {:0.2e}, best iter: {}, terminal leaves: {}'.format(iter, time()-t0,
                                                                                           se + regularization, se,
                                                                                           regularization, best_iter,
                                                                                           terminal_leaves))
            loss = se + regularization

            if loss < lowest_loss:
                best_iter = iter
                lowest_loss = np.copy(loss)
            if iter - best_iter > self.early_stopping_rounds:
                break

            neg_grad, hessian = self._get_neg_grad_and_hessian_diags(tree, y_tr, y_hat, iter+1, x)
            if do_plot:
                if type(tree.loss) in [QuantileLoss, QuadraticQuantileLoss]:
                    ax.cla()
                    n_q = y_hat.shape[1]
                    n_plot = 200
                    colors = plt.get_cmap('plasma', int(n_q))
                    for fl in np.arange(np.floor(n_q / 2), dtype=int):
                        q_low = np.squeeze(y_hat[:n_plot, fl])
                        q_up = np.squeeze(y_hat[:n_plot, n_q - fl - 1])
                        x_plot= np.arange(len(q_low))
                        ax.fill_between(x_plot, q_low, q_up, color=colors(fl), alpha=0.1 + 0.6 * fl / n_q,
                                        linewidth=0.0)

                    ax.plot(y[:n_plot, :], linewidth=2, label='target')
                    ax.legend(loc='upper right')
                    plt.title('Quantiles on first {} samples'.format(n_plot))
                else:
                    ax.cla()
                    ax.plot([np.min(y_tr[:, 0]), np.max(y_tr[:, 0])], [np.min(y_tr[:, 0]), np.max(y_tr[:, 0])], '--')
                    ax.scatter(y_tr[:, 0], y_hat[:, 0], marker='.', alpha=0.2)
                    ax.set_xlabel('observations')
                    ax.set_ylabel('predictions')
                    ax.set_title('Fit on first y dimension')

                plt.pause(0.1)

        print('#---------------- Model fitted in {:0.2e} min ----------------'.format((time()-t_init)/60))

        # keep trees up to best iteration
        self.trees = self.trees[:best_iter+1]
        if do_plot:
            plt.close(fig)

        return self

    def predict(self, x, n=None, x_lr=None):
        """
        Predicts the target based on the feature matrix x (and linear regression features x_lr).

        :param x: feature matrix, np.ndarray.
        :param n: predict up to the nth fitted tree. If None, predict all the trees. Default: None
        :param x_lr: linear regression feature matrix, np.ndarray. Only required if LinearLoss has been used.

        :return: target's predictions
        """
        y_hat = self.y_0
        if n is None:
            for tree in self.trees:
                y_hat = y_hat + tree.predict(x, x_lr)
        else:
            for tree in self.trees[:n]:
                y_hat = y_hat + tree.predict(x, x_lr)
        return y_hat

    def _get_neg_grad_and_hessian_diags(self, tree, y: np.ndarray, y_hat: np.ndarray, iteration: int, x: np.ndarray):
        """
        Returns the negative gradient at current iteration for all the observations, and a matrix which ith row is
        the diagonal of the Hessian for the current observation. This matrix is later used by the loss functions to
        obtain the inverse of the Hessian in each leaf, with proper dimensions.

        :param tree: current fitted tree
        :param y: targets
        :param y_hat: predictions at current iteration
        :param iteration: current iteration
        :param x: feature treaning set (for finding tree regions, if needed by the loss,
        as for the quadratic_quantile loss)

        :return: negative gradient and Hessian's diags
        """
        if iteration == 0:
            leaves_idx = [np.ones(len(y), dtype=bool)]
        else:
            leaves_idx = tree._find_regions(x, y)

        grad, hessian_diags = tree.loss.get_grad_and_hessian_diags(y, y_hat, iteration, leaves_idx)

        return - grad, hessian_diags

    def _fit_initial_guess(self, tree, y) -> np.ndarray:
        if type(tree.loss) in [QuantileLoss, QuadraticQuantileLoss]:
            _tree_pars = deepcopy(self.tree_pars)
            _tree_pars.update({"loss_type":"mse"})
            _tree_pars.update({"n_boosts":self.n_boosts})
            self.detrender = MBT(**_tree_pars).fit(x, y)
            preds = self.detrender.predict(x)
            err = y - preds
            qs = np.quantile(err, self.tree_pars['alphas'], axis=0)
            y_0 = np.squeeze(preds[:, :, None] + qs.T)
            self.qs_0 = qs
            self.y_0 = y_0
        else:
            y_0 = tree.loss.get_initial_guess(y)
            self.y_0 = y_0
        return y_0

    def _validation_split(self,x,y,x_lr):
        if self.val_ratio == 0:
            x_tr = x_val = x
            y_tr = y_val = y
            x_lr_tr = x_lr_val = x_lr
        else:
            randidx = np.random.permutation(x.shape[0])
            n_val = int(self.val_ratio*x.shape[0])
            tr_idx, val_idx = [randidx[:-n_val],randidx[-n_val:]]
            x_tr, x_val = [x[tr_idx],x[val_idx]]
            y_tr, y_val = [y[tr_idx], y[val_idx]]
            x_lr_tr, x_lr_val = [x_lr[tr_idx], x_lr[val_idx]]

        return x_tr, x_val, y_tr, y_val, x_lr_tr, x_lr_val


if __name__ == '__main__':
    pass