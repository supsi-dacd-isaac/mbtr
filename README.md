![description](docs/build/html/_static/logo.svg) 
# Multivariate Boosted Tree

## What is MBT
MBT is a python package for multivariate boosted trees trained in parameter space. 
The package can handle arbitrary multivariate losses, as long as its gradient and Hessian are known.
Gradient boosted trees are competition-winning, general-purpose, non-parametric regressors, which exploit sequential model fitting
and gradient descent to minimize a specific loss function. The most popular implementations are tailored to univariate regression and
classification tasks, precluding the possibility of capturing multivariate target cross-correlations and applying conditional penalties
to the predictions. The package allows to arbitrarily regularize the predictions, so that properties like smoothness, consistency and functional relations can be
enforced.

