import json
import numpy as np
import os
import pickle as pk
import requests
import lightgbm as lgb
import tqdm
import pandas as pd
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt

PATH_POWER = 'power_data.p'


def check_pars(required_pars, **kwargs):
    assert np.all([p in kwargs.keys() for p in required_pars]), 'Some of' \
                                                                ' this loss required parameters are' \
                                                                ' missing: {}'.format(required_pars)


def load_dataset():
    if os.path.exists(PATH_POWER):
        power_data = pk.load(open(PATH_POWER, "rb"))
    else:
        power_data = download_dataset()
    return power_data


def download_dataset():
    power = requests.get('https://zenodo.org/record/3463137/files/power_data.p?download=1')
    with open(PATH_POWER, "wb") as f:
        f.write(power.content)
    power_data = pk.load(open(PATH_POWER, "rb"))
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


def set_figure(size, subplots=(1,1), context='paper', style='darkgrid',font_scale = 1, l=0.2, w=0.1, h=0.1, b=0.1):
    sb.set(context=context,style=style, font_scale=font_scale)
    fig, ax = plt.subplots(subplots[0],subplots[1], figsize=size)
    plt.subplots_adjust(left=l, wspace=w, hspace=h, bottom=b)
    return fig, ax