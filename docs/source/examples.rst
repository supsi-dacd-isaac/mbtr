Examples
========

Offline synthetic example
-------------------------

Use this example to verify an installation without downloading data or opening
plots:

.. code-block:: python

   import numpy as np
   from mbtr import MBTRegressor

   rng = np.random.default_rng(2)
   X = rng.normal(size=(500, 5))
   Y = np.column_stack(
       (
           1.5 * X[:, 0] - X[:, 1],
           X[:, 0] + X[:, 2] - 0.5 * X[:, 3],
       )
   )

   model = MBTRegressor(
       n_estimators=40,
       max_depth=4,
       min_samples_leaf=12,
       random_state=2,
   )
   model.fit(X[:400], Y[:400])
   prediction = model.predict(X[400:])

External dataset examples
-------------------------

The scripts in ``examples/`` use the `Hierarchical Demand Forecasting
Benchmark <https://zenodo.org/record/3463137>`_.  They are optional integration
examples, not installation smoke tests.  Install their dependencies with:

.. code-block:: bash

   python -m pip install "mbtr[plot,examples]"

The upstream artifact is a legacy pickle.  Load it only if you trust its
source.  The downloader uses a timeout and HTTP status checks, but pickle does
not provide a safe untrusted-data boundary.

``multivariate_forecast.py``
   Compares an ordinary multivariate tree, a linear-response tree, and an
   optional independent LightGBM baseline.

``fourier_and_smoothing.py``
   Compares second-difference smoothing with Fourier response bases.

``quantiles.py``
   Fits ordered quantile levels and demonstrates the explicit
   ``(sample, output, quantile)`` prediction shape.

Linear-response outline
-----------------------

.. code-block:: python

   model = MBTRegressor(
       objective="linear_regression",
       n_estimators=30,
       l2_regularization=1e-3,
   )
   model.fit(
       split_features_train,
       target_train,
       response_features=response_features_train,
   )
   prediction = model.predict(
       split_features_test,
       response_features=response_features_test,
   )

Quantile outline
----------------

.. code-block:: python

   levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
   model = MBTRegressor(
       objective="quantile",
       objective_params={"alphas": levels},
       refit_leaves=True,  # empirical residual quantiles; this is the default
   )
   model.fit(X_train, Y_train)
   quantile_prediction = model.predict(X_test)

   # Single-output plotting convenience: (sample, quantile)
   single_output_quantiles = quantile_prediction[:, 0, :]

The quantile example grows trees with the smooth logistic construction
surrogate but reports pinball loss and refits leaf responses to empirical
residual quantiles.
