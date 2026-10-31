Extending MBTR
==============

Objective contract
------------------

An objective is a :class:`mbtr.losses.Loss` instance.  During structure search,
the central invariant is that ``solve_leaf`` returns both a candidate response
and the gain achieved by that response under the same local approximation.
This keeps split comparisons and their stored construction responses
mathematically aligned.

The construction surrogate, public statistical loss, and final leaf response
need not always be identical.  The published quantile objectives deliberately
use smooth derivatives for split search and then optionally replace terminal
responses with exact pinball minimizers.  Such a post-structure refit must be
explicit through ``supports_leaf_refit`` and ``refit_leaf``; ``loss`` must
remain the metric intended for histories, early stopping, and backtracking.

The main hooks are:

``configure(n_outputs)``
   Validate output-dependent parameters and build reusable matrices or bases.

``prepare_target(y)``
   Convert the public target to the internal two-dimensional representation.

``initial_prediction(y)``
   Return one internal prediction vector used before the first tree.

``gradient_hessian(y, prediction)``
   Return the negative gradient and nonnegative curvature arrays for an
   ordinary objective.

``partitioned_gradient_hessian(y, prediction, partitions)``
   Optional construction hook for an objective with
   ``requires_partitions = True``.  ``partitions`` contains the preceding
   tree's terminal labels; the first round uses one global partition.  The
   base implementation delegates to ``gradient_hessian``.

``solve_leaf(gradient_sum, hessian_sum, count, ...)``
   Return :class:`mbtr.losses.LeafSolution`, containing the optimal value and
   its positive gain.

``loss(y, prediction)``
   Return the scalar metric used for histories and early stopping.

``restore_prediction(prediction)``
   Restore public output axes, as done for quantiles.

Objectives requiring empirical terminal-leaf refitting set
``supports_leaf_refit = True`` and implement ``refit_leaf``.  Surrogate-only
objectives may also use this hook for an exact minimizer of their own smooth
loss.

Verification checklist
----------------------

Before exposing an objective, add tests that:

#. compare its response with an analytical or direct numerical minimizer;
#. verify the stationarity equation :math:`A\theta=b`;
#. verify the reported gain equals the direct quadratic decrease;
#. check gradients and curvature with central finite differences, holding any
   required partition labels fixed;
#. test extreme values, scalar outputs, and float32/float64 inputs;
#. compare histogram aggregation with a slow row-filtered implementation;
#. prove any response constraint, such as membership in a basis span;
#. prove that an exact refit does not increase its declared statistical loss;
#. for quantile objectives, test pinball score and calibration on data with
   known conditional quantiles;
#. for classification objectives, test high-margin losses and derivatives
   beyond the floating-point precision cliff rather than relying only on
   moderate-logit finite differences;
#. validate every public output shape and malformed-input error.

Multiclass objectives
---------------------

Grouped softmax is not a diagonal objective.  An exact implementation must
retain the per-target block Hessian, preserve invariance to a common logit
shift, and produce gradients whose entries sum to zero.  In particular, do
not form the target gradient as ``softmax(logits)[target] - 1`` at large
margins: that subtraction can round to zero while non-target gradients remain
nonzero.  Follow the target-excluded formulation in `A Fix for Softmax and
Cross-Entropy <https://arxiv.org/abs/2610.31642>`_ and add shift-invariance,
zero-sum, positive-semidefinite block-Hessian, and high-margin reference tests
before registering a softmax objective.

Backend boundary
----------------

Backends accelerate aggregation, partitioning, or routing; they do not define
objective semantics.  A backend receives contiguous numeric arrays and emits
backend-neutral statistics or compact tree arrays.  Device-specific arrays
must not leak through :class:`mbtr.MBTRegressor` or serialized model state.

A new backend needs:

* split-objective parity with the NumPy engine for identical bins, allowing a
  different winner only for numerically tied gains;
* prediction/objective parity under dtype-specific tolerances;
* cold and warm benchmark results;
* peak-memory and packaging measurements;
* a documented intended workload and break-even point;
* a fallback that loads the serialized model without the accelerator.

Tree visualization
------------------

Core trees use compact numeric nodes.  NetworkX is an optional export for
inspection and plotting, not a fitting or prediction data structure.  Keep
visualization dependencies outside core imports.
