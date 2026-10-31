User guide
==========

Basic regression
----------------

The following example is offline and deterministic:

.. doctest::

   >>> import numpy as np
   >>> from mbtr import MBTRegressor
   >>> rng = np.random.default_rng(11)
   >>> X = rng.normal(size=(240, 4))
   >>> Y = np.column_stack((2 * X[:, 0] - X[:, 1], X[:, 0] + X[:, 2]))
   >>> model = MBTRegressor(
   ...     n_estimators=12,
   ...     max_bins=16,
   ...     min_samples_leaf=8,
   ...     max_depth=3,
   ...     random_state=11,
   ... )
   >>> _ = model.fit(X[:200], Y[:200])
   >>> model.predict(X[200:]).shape
   (40, 2)

Targets may be one- or two-dimensional.  Ordinary one-dimensional targets are
returned as one-dimensional predictions.  Structured quantile objectives use
the explicit shape described below.

Binary and multilabel classification
------------------------------------

``MBTClassifier`` implements hard-target Bernoulli classification with shared
tree partitions.  A one-dimensional target represents ordinary binary
classification; a two-dimensional target is interpreted as a multilabel
indicator matrix.  In both cases every target value must be exactly ``0`` or
``1``.  With the default constant responses, each terminal leaf stores a logit
increment for every label.  The corresponding binary response is constant only
after applying the sigmoid and ``decision_threshold``; the leaf itself does not
store a hard class.

.. doctest::

   >>> import numpy as np
   >>> from mbtr import MBTClassifier
   >>> rng = np.random.default_rng(12)
   >>> X = rng.normal(size=(240, 4))
   >>> Y = np.column_stack((X[:, 0] + X[:, 1] > 0, X[:, 2] - X[:, 3] > 0))
   >>> model = MBTClassifier(
   ...     n_estimators=12,
   ...     max_bins=16,
   ...     min_samples_leaf=8,
   ...     max_depth=3,
   ...     random_state=12,
   ... )
   >>> _ = model.fit(X[:200], Y[:200])
   >>> model.decision_function(X[200:]).shape
   (40, 2)
   >>> model.predict_proba(X[200:]).shape
   (40, 2)
   >>> model.predict(X[200:]).shape
   (40, 2)

For multilabel data, ``predict_proba`` contains the positive-class probability
for each label.  For a one-dimensional binary target, it instead returns the
usual two columns for classes zero and one.  ``predict`` applies the estimator's
``decision_threshold`` and ``score`` returns exact-match accuracy (also called
subset accuracy for multilabel data).

Training, validation, early stopping, and monotonic step acceptance all use
mean binary cross-entropy evaluated on raw logits.  The implementation uses
label-conditioned sigmoid tails for the derivatives and ``logaddexp`` for the
loss.  This avoids the high-margin cancellation that would occur in
``y - sigmoid(logit)`` and follows the numerical principle described in
`A Fix for Softmax and Cross-Entropy <https://arxiv.org/abs/2610.31642>`_.

This release does not interpret one-hot matrices as multiclass targets and does
not use softmax.  Exact grouped softmax requires block Hessians and a
target-excluded, zero-sum gradient construction; it will be added separately
rather than hidden behind a diagonal approximation.

Linear-logistic responses
~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``response_features`` to fit a ridge-linear logit function inside every
leaf.  The original feature matrix still determines the tree partitions and
routing; the response-feature matrix is used only by the leaf models.  An
intercept is appended internally.

.. code-block:: python

   model = MBTClassifier(
       n_estimators=50,
       l2_regularization=1e-3,
       linear_response_curvature="upper_bound",
       random_state=12,
   )
   model.fit(
       split_features_train,
       Y_train,
       response_features=response_features_train,
       eval_set=(
           split_features_validation,
           Y_validation,
           response_features_validation,
       ),
   )
   probability = model.predict_proba(
       split_features_test,
       response_features=response_features_test,
   )

The response design is required by ``decision_function``, ``predict_proba``,
``predict``, ``score``, and staged prediction after a linear-response fit.  A
classifier fitted without it deliberately rejects a response design at
prediction time, preventing an accidental change of model semantics.

``linear_response_curvature="upper_bound"`` is the default.  It uses the
global Bernoulli curvature bound :math:`1/4`, so every label shares a single
:math:`Z^T Z` Gram matrix.  ``"exact"`` instead accumulates
:math:`Z^T\operatorname{diag}(p_j(1-p_j))Z` separately for each label
:math:`j`.  Exact curvature can better represent the local Newton geometry,
but its Gram storage, accumulation, and solves scale with the label count and
the square of the response-design width.  The upper-bound mode has the same
matrix order as linear-response regression and is the practical default for
wide multilabel problems.  ``regularize_intercept=False`` leaves the appended
intercept outside the ridge penalty; set it to ``True`` explicitly to change
that policy.

Validation and early stopping
-----------------------------

Prefer an explicit validation set:

.. code-block:: python

   model = MBTRegressor(
       n_estimators=300,
       early_stopping_rounds=20,
       random_state=4,
   )
   model.fit(X_train, Y_train, eval_set=(X_validation, Y_validation))

   print(model.best_iteration_)
   print(model.best_score_)
   print(model.evals_result_["training"])
   print(model.evals_result_["validation"])

``validation_fraction`` is a convenience random split controlled by
``random_state``.  Random splitting may leak future information in forecasting
and other ordered data.  In those cases, build a chronological ``eval_set``.

Objectives
----------

Objective-specific values are supplied through ``objective_params``.

.. list-table:: Supported objectives
   :header-rows: 1
   :widths: 23 32 45

   * - Name
     - Required values
     - Response
   * - ``squared_error`` or ``mse``
     - none
     - Independent constant response per output.
   * - ``time_smoother``
     - ``lambda_smooth``
     - Constant vector with a second-difference penalty.
   * - ``fourier``
     - ``n_harmonics``
     - Response in a non-redundant Fourier subspace.
   * - ``latent_variable``
     - ``S`` and ``precision``
     - Response constrained to the column space of ``S``.
   * - ``linear_regression``
     - ``response_features`` passed to fitting and prediction
     - Ridge-linear function inside each leaf.
   * - ``quantile``
     - ordered ``alphas``; optional positive ``smoothing``
     - Logistic split surrogate with empirical-quantile refitting and pinball monitoring.
   * - ``quadratic_quantile``
     - ordered ``alphas``; optional ``quadratic_scale`` and ``denominator_floor``
     - Published partition-dependent linear--quadratic construction with empirical refitting.
   * - ``pseudo_huber_quantile``
     - ordered ``alphas``; optional positive ``delta``
     - Convex asymmetric pseudo-Huber objective optimized end to end.

Smoothing
~~~~~~~~~

.. code-block:: python

   model = MBTRegressor(
       objective="time_smoother",
       objective_params={"lambda_smooth": 2.0},
       l2_regularization=1e-3,
   )
   model.fit(X_train, Y_train)

The smoothing penalty is applied to each leaf update, not to the final sum of
all trees.  Penalizing the final ensemble would introduce different cross-tree
terms and is a different objective.

Fourier responses
~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = MBTRegressor(
       objective="fourier",
       objective_params={"n_harmonics": 3},
   )

``n_harmonics`` counts positive frequencies; the constant component is added
once automatically.  The implementation removes degenerate zero and duplicate
basis vectors and solves the regularized Gram system.  It does not assume every
raw sine/cosine column has the same norm.

Latent responses
~~~~~~~~~~~~~~~~

For ``S`` with shape ``(n_outputs, n_latent)`` and a symmetric positive
semidefinite precision matrix:

.. code-block:: python

   model = MBTRegressor(
       objective="latent_variable",
       objective_params={"S": S, "precision": precision},
   )

Every leaf response lies in the column space of ``S``.

Linear regression responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Split features and response features have separate roles:

.. code-block:: python

   model = MBTRegressor(
       objective="linear_regression",
       l2_regularization=1e-3,
   )
   model.fit(
       split_features_train,
       Y_train,
       response_features=response_features_train,
       eval_set=(
           split_features_validation,
           Y_validation,
           response_features_validation,
       ),
   )
   prediction = model.predict(
       split_features_test,
       response_features=response_features_test,
   )

The intercept is not regularized by default.  Set
``objective_params={"regularize_intercept": True}`` to change that policy.

Quantiles
~~~~~~~~~

.. code-block:: python

   levels = np.array([0.1, 0.5, 0.9])
   model = MBTRegressor(
       objective="quantile",
       objective_params={"alphas": levels, "smoothing": 0.5},
   )
   model.fit(X_train, Y_train)
   quantiles = model.predict(X_test)

``quantiles`` has shape ``(n_samples, n_outputs, n_quantiles)`` even when
``n_outputs == 1``.  Quantile levels must be strictly increasing.  The model
does not silently sort levels or hide quantile crossings.

The default ``quantile`` behavior follows the published MBTR method.  It uses
the logistic-smoothed negative gradient and curvature to select splits, then
replaces each terminal response with the empirical quantile of the residuals
in that leaf.  The initial prediction is empirical as well.  Consequently,
``evals_result_``, validation-based early stopping, backtracking, and
``score`` use pinball loss, which is the statistical objective, rather than
the construction surrogate.  Set ``refit_leaves=False`` to keep the
second-order responses produced during split construction.

``quadratic_quantile`` implements the paper's partition-dependent
linear--quadratic relaxation.  ``quadratic_scale`` defaults to ``1e5`` and
``denominator_floor`` prevents unstable division when a signed residual sum is
near zero.  Its curvature is stabilized to remain positive for tree fitting;
like ``quantile``, it reports pinball loss and uses empirical residual-quantile
refitting by default.

``pseudo_huber_quantile`` is a distinct, fully smooth alternative.  Its
``delta`` controls the transition between locally quadratic and approximately
linear behavior.  It keeps surrogate-optimal initialization, refitting, and
monitoring, so it should not be interpreted as the paper's
``quadratic_quantile`` objective.

Tree controls
-------------

``min_samples_leaf`` applies to both children.  ``max_depth`` and
``max_leaves`` limit tree size, while ``min_split_gain`` requires a positive
regularized improvement.  Global quantile bins are built once from the
training set using ``max_bins`` and reused across boosting rounds.

Fitted attributes and methods
-----------------------------

Useful fitted attributes include ``best_iteration_``, ``best_score_``,
``n_features_in_``, ``n_outputs_``, ``feature_importances_``, and
``evals_result_``.  Use ``staged_predict`` for predictions after each boosting
round and ``apply`` for terminal leaf identifiers.  Classifiers additionally
provide ``classes_``, ``n_classes_``, ``n_labels_``,
``staged_decision_function``, and ``staged_predict_proba``.
For a one-dimensional binary target, ``classes_`` is ``[0, 1]`` and
``n_classes_`` is ``2``.  For a two-dimensional multilabel target, these are
per-label metadata (a list of class arrays and an integer array,
respectively), matching scikit-learn's multi-output convention.
