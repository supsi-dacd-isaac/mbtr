<img src="docs/source/_static/logo.svg"> 

[![Documentation Status](https://readthedocs.org/projects/mbtr/badge/?version=master)](https://mbtr.readthedocs.io/en/master/?badge=master)
[![Build Status](https://travis-ci.org/supsi-dacd-isaac/mbtr.svg?branch=master)](https://travis-ci.org/supsi-dacd-isaac/mbtr)
[![codecov](https://codecov.io/gh/supsi-dacd-isaac/mbtr/branch/master/graph/badge.svg)](https://codecov.io/gh/supsi-dacd-isaac/mbtr)
[![Latest Version](https://img.shields.io/pypi/v/mbtr.svg)](https://pypi.python.org/pypi/mbtr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# Multivariate Boosted TRee

## What is MBTR

MBTR is a python package for multivariate boosted tree regressors trained in parameter space. 
The package can handle arbitrary multivariate losses, as long as their gradient and Hessian are known.
Gradient boosted trees are competition-winning, general-purpose, non-parametric regressors, which exploit sequential model fitting and gradient descent to minimize a specific loss function. The most popular implementations are tailored to univariate regression and classification tasks, precluding the possibility of capturing multivariate target cross-correlations and applying conditional penalties to the predictions. This package allows to arbitrarily regularize the predictions, so that properties like smoothness, consistency and functional relations can be
enforced.

## Installation

```sh
pip install --upgrade git+https://github.com/supsi-dacd-isaac/mbtr.git
```

## Usage 

MBT regressor follows the scikit-learn syntax for regressors. Creating a default instance and training it is as simple as:
```python
m = MBT().fit(x,y)
```
while predictions for the test set are obtained through 

```python
y_hat = m.predict(x_te)
```
The most important parameters are the number of boosts `n_boost`, that is, the number of fitted trees, `learning_rate` and the `loss_type`. An extensive explanation of the different parameters can be found in the documentation. 



## Documentation 

Documentation and examples on the usage can be found at [docs](https://mbtr.readthedocs.io/en/master/?badge=master).

## Reference

If you make use of this software for your work, we would appreciate it if you would cite us:

*Lorenzo Nespoli and Vasco Medici (2022).
Multivariate Boosted Trees and Applications to Forecasting and Control*
[JMLR paper](https://www.jmlr.org/papers/volume23/21-0247/21-0247.pdf)

```
@article{JMLR:v23:21-0247,
  author  = {Lorenzo Nespoli and Vasco Medici},
  title   = {Multivariate Boosted Trees and Applications to Forecasting and Control},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {246},
  pages   = {1--47},
  url     = {http://jmlr.org/papers/v23/21-0247.html}
}
```

## Acknowledgments

The authors would like to thank the Swiss Federal Office of Energy (SFOE) and the
Swiss Competence Center for Energy Research - Future Swiss Electrical Infrastructure (SCCER-FURIES),
for their financial and technical support to this research work.
