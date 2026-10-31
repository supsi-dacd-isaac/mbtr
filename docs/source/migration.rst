Migrating from v0.1
===================

v0.2 corrects mathematical and implementation defects, so retraining can
produce different trees and predictions.  In particular, it preserves signed
gradients for coupled objectives, enforces minimum child sizes, makes split
partition semantics consistent, repairs validation, and stabilizes quantile
curvature without replacing the published hybrid fitting strategy.

Name changes
------------

.. list-table:: Parameter migration
   :header-rows: 1

   * - v0.1
     - v0.2
   * - ``MBT``
     - ``MBTRegressor``
   * - ``n_boosts``
     - ``n_estimators``
   * - ``loss_type``
     - ``objective``
   * - loss-specific ``**kwargs``
     - ``objective_params={...}``
   * - ``n_q``
     - ``max_bins``
   * - ``min_leaf``
     - ``min_samples_leaf``
   * - ``lambda_weights``
     - ``l2_regularization``
   * - ``lambda_leaves``
     - ``leaf_regularization``
   * - ``x_lr``
     - ``response_features``
   * - ``val_ratio``
     - explicit ``eval_set`` or ``validation_fraction``
   * - ``refit``
     - ``refit_leaves``

Before:

.. code-block:: python

   from mbtr.mbtr import MBT

   model = MBT(
       loss_type="time_smoother",
       lambda_smooth=2.0,
       n_boosts=50,
       n_q=16,
       min_leaf=20,
   )
   model.fit(X_train, Y_train, do_plot=True)

After:

.. code-block:: python

   from mbtr import MBTRegressor

   model = MBTRegressor(
       objective="time_smoother",
       objective_params={"lambda_smooth": 2.0},
       n_estimators=50,
       max_bins=16,
       min_samples_leaf=20,
   )
   model.fit(X_train, Y_train)

Compatibility window
--------------------

``MBT`` and the old argument names remain available through a compatibility
wrapper for one transition cycle.  They should not be used in new examples or
saved model pipelines.  ``Tree`` and the mutable loss-name registry are now
implementation-level interfaces.

Behavior changes
----------------

Validation
~~~~~~~~~~

The recommended interface is ``eval_set=(X_valid, Y_valid)``.  A
linear-response validation set includes its response features as the third
item.  ``validation_fraction`` is deterministic only when ``random_state`` is
set and should not be used for time-ordered forecasting.

Progress and plotting
~~~~~~~~~~~~~~~~~~~~~

Fitting no longer starts plotting or prints elapsed time unconditionally.
Use ``verbose``, callbacks, and ``evals_result_``.  Plotting stays outside the
estimator and is therefore safe in headless environments.

Quantile shape
~~~~~~~~~~~~~~

v0.1 often squeezed a single-output quantile prediction to
``(samples, quantiles)``.  v0.2 consistently returns
``(samples, outputs, quantiles)``.  For a single output, select
``prediction[:, 0, :]`` explicitly.

Quantile objective semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The historical ``quantile`` behavior is retained deliberately: logistic
derivatives construct the tree, empirical residual quantiles replace terminal
responses when ``refit_leaves=True``, and pinball loss is used for reporting,
validation, and step acceptance.  This is the hybrid method described in the
MBTR publication, not a derivative inconsistency.

Likewise, ``quadratic_quantile`` denotes the published partition-dependent
linear--quadratic formulation.  Its optional stability parameters are
``quadratic_scale`` (default ``1e5``) and ``denominator_floor``.  Existing
v0.1 code using this objective keeps the same conceptual model, with safer
curvature handling.

The asymmetric pseudo-Huber formulation is now explicitly named
``pseudo_huber_quantile`` and accepts ``delta``.  Code written against a
development version that passed ``delta`` to ``quadratic_quantile`` should
migrate as follows:

.. code-block:: python

   model = MBTRegressor(
       objective="pseudo_huber_quantile",
       objective_params={"alphas": levels, "delta": 0.5},
   )

Unlike the two published hybrid objectives, the pseudo-Huber variant uses its
smooth surrogate for initialization, leaf optimization, and monitoring.

Serialization
~~~~~~~~~~~~~

v0.1 pickles captured NetworkX and implementation details and are not a stable
v0.2 interchange format.  Refit old models under v0.2.  Newly serialized
models carry a format version and reject incompatible state with an explicit
error rather than silently loading it incorrectly.
