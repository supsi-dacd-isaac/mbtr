import json
import numpy as np
import os
import pickle as pk
import requests
import lightgbm as lgb
import tqdm
import pandas as pd
from pathlib import Path


def check_pars(required_pars, **kwargs):
    assert np.all([p in kwargs.keys() for p in required_pars]), 'Some of' \
                                                                ' this loss required parameters are' \
                                                                ' missing: {}'.format(required_pars)


def download_dataset():
    rel_path = os.path.dirname(__file__)
    path_power = os.path.join(rel_path, r"../data/power_data.p")
    try:
        power_data = pk.load(open(path_power, "rb"))
    except:
        Path("../data").mkdir(parents=True, exist_ok=True)
        print('#' * 20 + '    Downloading example dataset    ' + '#' * 20)
        power = requests.get('https://zenodo.org/record/3463137/files/power_data.p?download=1')
        with open(path_power, "wb") as f:
            f.write(power.content)
        power_data = pk.load(open(path_power, "rb"))
    return power_data


class LightGBMMISO:
    def __init__(self, n_estimators, lgb_pars = None):
        self.n_estimators = n_estimators
        if lgb_pars is None:
            self.lgb_pars = {"objective": "regression",
                             "max_depth": 20,
                             "num_leaves": 100,
                             "learning_rate": 0.1,
                             "verbose": -1,
                             "metric": "l2",
                             "min_data": 4,
                             "num_threads":4}
        else:
            self.lgb_pars = lgb_pars
        self.m = []

    def fit(self, x, y):

        for i in tqdm.tqdm(range(y.shape[1])):
            lgb_train = lgb.Dataset(x, y[:, i].ravel())
            self.m.append(lgb.train(self.lgb_pars, lgb_train, num_boost_round=self.n_estimators))
        return self

    def predict(self,x):
        y = []
        for m in self.m:
            y.append(m.predict(x).reshape(-1,1))
        y = np.hstack(y)
        return y
