Performance and benchmarks
==========================

Optimization policy
-------------------

MBTR keeps the NumPy implementation as the correctness reference.  A faster
engine is accepted only after it matches the reference's sufficient
statistics, split objective, leaf responses, predictions, and end-to-end
metrics within dtype-specific tolerances.  Floating-point reduction order may
choose a different candidate when split gains are numerically tied; topology
and individual predictions are therefore not required to be bitwise identical.

The main algorithmic optimizations are independent of the implementation
language:

* bin features once rather than recomputing candidates in every node;
* accumulate signed gradient and curvature histograms together;
* evaluate candidates from prefix statistics;
* enforce child sizes before solving candidates;
* reuse row-index buffers rather than copying full matrices recursively;
* store nodes compactly and route prediction batches iteratively;
* use ``solve`` or Cholesky instead of explicit matrix inverses;
* for linear leaves, aggregate :math:`Z^T Z` and :math:`Z^T G` per bin;
* for exact linear-logistic curvature, aggregate the Hessian-weighted Gram
  separately for each label.
* batch every valid split candidate instead of dispatching one leaf solve per
  threshold;
* retain raw node histograms, build only the smaller child, and derive its
  sibling by histogram subtraction when the cache fits a 256 MiB guard;
* derive squared-error Hessian statistics directly from counts;
* build indexed constant-Hessian histograms without materializing node-sized
  feature and gradient copies;
* scan every feature's constant-Hessian candidates in one compiled pass;
* write training updates from the builder's leaf memberships instead of
  traversing each newly fitted tree again.

Backend status
--------------

``numpy``
   Authoritative engine and fallback.  It defines model semantics.

``numba``
   Verified optional CPU accelerator for ordinary and linear-response
   histogram and split-search kernels.  Backend tests require equivalent
   statistics, split objective, leaf values, predictions, and aggregate metrics
   within numerical tolerance. Near-tied split gains can produce a different
   valid topology because compiled and NumPy reductions need not round in the
   same order. ``engine="auto"`` selects Numba only for large ordinary or linear
   histogram workloads, while retaining NumPy for small workloads where import,
   JIT, and dispatch overhead would dominate. The automatic decision includes
   the number of boosting rounds because one compilation is amortized across
   every tree. ``engine="numba"`` is available when repeated warm fits make
   compilation cost less important.

Output sketching
----------------

``output_sketch_size=k`` optionally chooses tree structure from the ``k``
outputs with the largest gradient norms in each boosting round. Terminal leaf
values are then recomputed from every output, and full-output loss remains the
monotonic acceptance criterion. This is an approximate split-search strategy,
not a change to the reported objective. Keep it disabled for small output
dimensions and select ``k`` using validation data only.

The 100-tree RF1 and SCM20D gates found that halving 8 or 16 outputs worsened
RMSE by roughly 1.5--2.5 percent for only modest speed gains. The option is
therefore experimental and intended for datasets such as Mediamill and
Delicious, where output dimension is 101 or 983. It is never enabled by
``engine="auto"``.

Real-data optimization gate
---------------------------

The fixed-fold gate uses prepared RF1 and SCM20D fold 0, 100 trees, learning
rate 0.05, depth 4, 64 bins, and the response columns selected by the completed
leakage-safe standard fits. It warms Numba before timing and can compare current
predictions with a saved reference.

.. code-block:: bash

   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
   VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
   python -m benchmarks.bench_real_data_gate --n-estimators 100 --repeat 3

On the development Apple M5 machine, the original implementation's one-pass
times and the optimized implementation's three-pass medians were:

.. list-table:: Fixed-fold 100-tree fit times
   :header-rows: 1

   * - Dataset
     - Variant
     - Original seconds
     - Optimized median seconds
     - Speedup
     - RMSE before / after
   * - RF1
     - Standard
     - 18.89
     - 2.39
     - 7.9x
     - 4.777589 / 4.777508
   * - RF1
     - Linear
     - 72.28
     - 11.73
     - 6.2x
     - 2.652746 / 2.652805
   * - SCM20D
     - Standard
     - 28.93
     - 2.69
     - 10.8x
     - 151.890962 / 151.904868
   * - SCM20D
     - Linear
     - 100.68
     - 27.33
     - 3.7x
     - 111.232527 / 111.232527

These are local implementation gates, not cross-library benchmark claims. The
original column is a single pass, whereas the optimized column is a median, so
the speedups are deliberately reported only to two significant figures.

A later constant-Hessian pass used five repetitions at every decision point
and retained a change only when both RF1 and SCM20D became faster while test
RMSE remained effectively unchanged.  The resulting sequence was monotonic:

.. list-table:: Sequential 100-tree standard-MBTR gate
   :header-rows: 1

   * - Retained implementation
     - RF1 median seconds
     - SCM20D median seconds
   * - Starting point
     - 3.126
     - 3.541
   * - Avoid repeated ``einsum`` path planning
     - 2.512
     - 2.746
   * - Fuse all constant-Hessian split candidates in one compiled scan
     - 0.393
     - 0.695
   * - Build node histograms from row indices without materialized copies
     - 0.359
     - 0.607

This is an 8.7x RF1 speedup and a 5.8x SCM20D speedup over that pass's starting
point.  RF1 RMSE changed from 4.777508 to 4.770421 and SCM20D RMSE from
151.904868 to 151.904168.  A separate seven-repetition confirmation produced
0.436 and 0.686 second medians, respectively; the prediction hashes were stable
within each implementation.  Candidate-loss buffering, row-partition buffers,
histogram workspaces, and compiled validation traversal were measured and
rejected because at least one dataset became slower.

The same retained implementation was also run through the full five-fold,
1,000-round benchmark protocol with learning rate 0.05, validation patience 50,
and one native thread.  The table compares medians from that run with the
previous standard-MBTR and synchronized LightGBM artifacts:

.. list-table:: Five-fold 1,000-round fit-time check
   :header-rows: 1

   * - Dataset
     - Previous MBTR seconds
     - Current MBTR seconds
     - LightGBM seconds
     - Previous / current MBTR mean RMSE
   * - RF1
     - 26.595
     - 3.040
     - 3.734
     - 1.729184 / 1.727032
   * - SCM20D
     - 29.183
     - 4.991
     - 8.055
     - 103.838149 / 103.832204

These timings include validation monitoring.  They show that the earlier
LightGBM gap came primarily from Python/NumPy dispatch per feature and split,
not from an unavoidable cost of shared multivariate trees.  They do not imply
quality parity between the models: LightGBM still attained lower RMSE in these
fixed benchmark configurations.

Two Numba histogram threads improved the combined single-process gate by about
four percent; four threads added nothing. Keep ``NUMBA_NUM_THREADS=1`` when
running independent folds concurrently. For a single fit on an otherwise idle
machine, two threads are an optional measured setting.

Rust
   A later option for a profiler-confirmed stable CPU kernel.  Native packaging
   is justified only by a material end-to-end improvement over optimized
   NumPy/Numba code.

JAX
   An experimental option for large fixed-shape, breadth-first GPU workloads.
   Compile time, transfers, and synchronized execution must be reported
   separately.

KeOps
   Not currently planned: MBTR needs segmented histograms, prefix statistics,
   partitions, and small dense solves rather than implicit pairwise matrices.

Running ASV
-----------

Install the benchmark dependencies and run the quick local profile:

.. code-block:: bash

   python -m pip install -e ".[benchmark,fast]"
   asv run --quick

The scheduled profile uses the expanded matrix:

.. code-block:: bash

   MBTR_BENCHMARK_PROFILE=full asv run

The suite changes one dimension at a time around a representative baseline;
it deliberately avoids a full Cartesian product.

.. list-table:: Scaling axes
   :header-rows: 1

   * - Axis
     - Full values
   * - Training rows
     - 1k, 10k, 100k, 1m
   * - Split features
     - 4, 16, 64, 256
   * - Outputs
     - 1, 8, 32, 128
   * - Histogram bins
     - 16, 64, 256
   * - Linear-response features
     - 2, 8, 32
   * - Trees
     - 1, 10, 50
   * - Prediction batch
     - 1, 1k, 100k
   * - Dtype
     - float32, float64

Metrics include cold import, first compilation, warm kernels, one-tree fit,
ensemble fit, linear-response fit, prediction throughput, peak memory,
allocation-sensitive paths, leaf application, and serialized model size.
The estimator currently normalizes numeric inputs to float64, so the dtype axis
measures input-conversion cost rather than separate float32 kernel execution.

The dedicated linear-response scaling cases hold the tree topology at depth
three while varying rows, split features, outputs, bins, response-design width,
and tree count. They time both NumPy and warm Numba execution for regression
and linear-logistic classification and record peak memory along the
response-width axis. Classification's default ``upper_bound`` mode replaces
the Bernoulli curvature with its global :math:`1/4` bound and shares one Gram
matrix across labels, giving the same matrix order as linear-response
regression. Bounded cases also compare ``exact`` curvature, whose
Hessian-weighted Gram accumulation and solves scale with label count and
squared response-design width. This isolates matrix-size scaling from changes
in the number of fitted nodes without making the scheduled exact-curvature
matrix prohibitive.

Reproducibility
---------------

Performance results should record the commit, processor, memory, operating
system, Python, NumPy, SciPy, accelerator version, dtype, thread counts, and
whether execution was cold or warm.  Run on an otherwise idle controlled
machine and pin BLAS/accelerator thread counts.

Ordinary unit tests do not assert wall-clock durations.  A scheduled
performance job tracks history and investigates representative regressions of
roughly ten percent or more.  Scaling ratios are more portable than absolute
times: doubling rows or features should be approximately linear, and adding
bins must not multiply the row scan by the number of bins.

The scheduled GitHub-hosted run retains raw ASV results and an HTML report for
trend inspection, but shared-runner noise is not a release gate.  Confirm a
suspected regression on the same controlled machine before changing a backend
or accepting a new low-level implementation.
