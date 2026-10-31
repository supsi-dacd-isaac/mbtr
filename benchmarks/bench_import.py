"""Cold import and first-call compilation measurements."""

from __future__ import annotations

import os
import subprocess
import sys

from .common import make_model, make_regression


class TimeColdImport:
    number = 1
    repeat = 3
    timeout = 60

    def time_cold_import(self):
        environment = os.environ.copy()
        environment.setdefault("PYTHONHASHSEED", "0")
        subprocess.run(
            [sys.executable, "-c", "import mbtr; from mbtr import MBTRegressor"],
            check=True,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class TimeFirstAndWarmFit:
    params = ["numpy", "numba"]
    param_names = ["engine"]
    number = 1
    repeat = 1
    timeout = 120

    def setup(self, engine):
        self.x, self.y = make_regression(2_000, 16, 4)

    def time_first_fit(self, engine):
        make_model(n_estimators=1, engine=engine).fit(self.x, self.y)


class TimeWarmFit:
    params = ["numpy", "numba"]
    param_names = ["engine"]
    number = 1
    repeat = 3
    timeout = 120

    def setup(self, engine):
        self.x, self.y = make_regression(2_000, 16, 4)
        make_model(n_estimators=1, engine=engine).fit(self.x, self.y)

    def time_warm_fit(self, engine):
        make_model(n_estimators=1, engine=engine).fit(self.x, self.y)
