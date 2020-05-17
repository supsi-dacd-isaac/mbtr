import unittest
from mbtr.mbtr import MBT, Tree
from mbtr.losses import MSE
import numpy as np
import matplotlib.pyplot as plt
from mbtr import LOSS_MAP


class StandAloneTests(unittest.TestCase):
    def setUp(self) -> None:
        # create bivariate targets, linearly dependent on random data
        self.N = 5000
        np.random.seed(0)
        x = np.zeros((self.N, 5))
        x[:, [0]] = np.arange(self.N).reshape(-1, 1) / self.N
        x[:, [1]] = np.random.permutation(self.N).reshape(-1, 1) / self.N
        x[:, [2]] = (np.random.permutation(self.N).reshape(-1, 1) / self.N) ** 2
        x[:, [3]] = np.tanh(np.random.permutation(self.N).reshape(-1, 1) / self.N)
        x[:, [4]] = np.random.permutation(self.N).reshape(-1, 1) / self.N
        y1 = 1 * x[:, 0] + 2 * x[:, 1] + 3.5 * x[:, 2] + 4 * x[:, 3] + 5 * x[:, 4]
        y2 = 10 + 3 * x[:, 0] + 2.4 * x[:, 1] + 8.5 * x[:, 2] + 1.8 * x[:, 3] + 3 * x[:, 4]
        self.x = x
        self.y = np.hstack([y1.reshape(-1, 1), y2.reshape(-1, 1)])
        self.n_tr = int(self.N * 0.7)

    def test_loss_default_instantiation(self):
        for k, v in LOSS_MAP.items():
            pars = {k:np.eye(2) for k in v.required_pars}
            print('{}: {}'.format(k, v(**pars)))
        assert True

    def test_single_tree(self):
        n_tr = self.n_tr

        t = Tree()
        t.fit(self.x[:n_tr, :], self.y[:n_tr, :])

        # scatter results
        plt.figure()
        plt.scatter(self.y[n_tr:, 0], t.predict(self.x[n_tr:, :])[:, 0])
        plt.scatter(self.y[n_tr:, 1], t.predict(self.x[n_tr:, :])[:, 1])
        x_l = np.min(self.y[n_tr:])
        x_u = np.max(self.y[n_tr:])
        plt.plot([x_l, x_u], [x_l, x_u])
        plt.pause(1)
        plt.close('all')
        assert type(t.loss) in LOSS_MAP.values()

    def test_MSE_loss(self):
        n_tr = self.n_tr

        mbt = MBT()
        mbt.fit(self.x[:n_tr, :], self.y[:n_tr, :])
        y_hat = mbt.predict(self.x[n_tr:, :])

        plt.figure()
        plt.scatter(self.y[n_tr:, 0], y_hat[:, 0])
        plt.scatter(self.y[n_tr:, 1], y_hat[:, 1])
        x_l = np.min(self.y[n_tr:])
        x_u = np.max(self.y[n_tr:])
        plt.plot([x_l, x_u], [x_l, x_u])
        plt.pause(1)
        plt.close('all')
        assert len(mbt.trees) > 1

    def test_MBT_instantiations_own(self):
        mbt_pars = {"early_stopping_rounds": 2,
                    "n_boosts": 30,
                    "do_refit": True
                    }
        m = MBT(**mbt_pars)
        assert True

    def test_MBT_instantiations_tree(self):
        tree_pars = {"n_q": 4,
                     "min_leaf": 322}
        m = MBT(**tree_pars)
        assert True

    def test_MBT_instantiations_tree_loss_and_own(self):
        tree_pars = {"n_q": 4,
                     "min_leaf": 34}
        mbt_pars = {"early_stopping_rounds": 5,
                    "n_boosts": 32,
                    "do_refit": True
                    }
        loss_pars = {"lambda_weights": 0.1, "lambda_leaves": 0.1, "loss_type": "linear-regression"}
        pars = {**tree_pars, **mbt_pars, **loss_pars}
        m = MBT(**pars)
        assert True


if __name__ == '__main__':
    unittest.main()
