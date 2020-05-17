<img src="docs/source/_static/logo.svg"> 
# Multivariate Boosted Tree

## What is MBTR
MBTR is a python package for multivariate boosted tree regressors trained in parameter space. 
The package can handle arbitrary multivariate losses, as long as their gradient and Hessian are known.
Gradient boosted trees are competition-winning, general-purpose, non-parametric regressors, which exploit sequential model fitting and gradient descent to minimize a specific loss function. The most popular implementations are tailored to univariate regression and classification tasks, precluding the possibility of capturing multivariate target cross-correlations and applying conditional penalties to the predictions. This package allows to arbitrarily regularize the predictions, so that properties like smoothness, consistency and functional relations can be
enforced.

