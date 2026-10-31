Scientific studies
==================

The repository separates two kinds of evaluation:

* ``benchmarks/`` contains synthetic ASV performance regressions;
* ``studies/`` contains external datasets, scientific protocols, published
  reference values, and shared-versus-independent model comparisons.

This separation matters because a stable speed test is not evidence of
predictive quality, and a heterogeneous external dataset is not a suitable
wall-clock regression gate.

Evidence layers
---------------

The primary evidence must use genuinely multi-output targets.  The current
direct bridge is the SketchBoost suite (RF1, SCM20D, Mediamill, Delicious, and
MoA).  Selected Mulan protocols remain historical references, while the UCO /
KDIS multilabel sources and the ICML 2025 multi-output conformal-regression
corpus provide more modern expansion paths.  UCO 2024 and KDIS are separate
protocols: UCO ships ten train/generalization partitions, while the currently
implemented KDIS study uses fixed 2017 iterative-stratified five-fold archives.
The MOCR corpus generates splits from seeds rather than shipping frozen
indices; the study catalog pins the upstream revision and requires generated
indices to be persisted.

TALENT and OpenML are complementary scalar studies.  TALENT-tiny offers 15
binary classification and 18 regression tasks supported by MBTR today.
OpenML-CC18 contributes 35 supported binary tasks, and OpenML-CTR23 contributes
35 regression tasks with public baselines.  None of these scalar tasks can
measure the benefit of sharing tree partitions across outputs.

Runnable OpenML pipeline
------------------------

Install the optional client dependencies from a repository checkout:

.. code-block:: bash

   python -m pip install -e '.[studies]'

The OpenML workflow is split into independently resumable stages:

.. code-block:: bash

   python -m studies.openml.pipeline prepare --suite ctr23 --workspace studies/data/openml-ctr23
   python -m studies.openml.pipeline run --workspace studies/data/openml-ctr23
   python -m studies.openml.pipeline aggregate --workspace studies/data/openml-ctr23
   python -m studies.openml.pipeline fetch-scores --workspace studies/data/openml-ctr23
   python -m studies.openml.pipeline report --workspace studies/data/openml-ctr23

For a small end-to-end installation check, run:

.. code-block:: bash

   python -m studies.openml.pipeline all \
     --suite ctr23 \
     --workspace studies/data/openml-ctr23-smoke \
     --max-tasks 1 \
     --max-runs-per-task 2 \
     --n-estimators 10 \
     --engine numpy

This intentionally incomplete workload is a smoke test, not a score
comparison.  The complete CTR23 suite has 800 outer resamples; omit both
selection limits to run it.  CC18 preparation runs only its 35 binary tasks and
records multiclass tasks as unsupported.

The suite lock records the exact task list, metadata, preprocessing policy, and
failures, while a run lock records model settings and the relevant source-tree
fingerprint.  Separate directories preserve official indices, prepared folds,
per-resample results, task aggregates, and verified reference sources.  Exact
fingerprint matches resume; selection, code, data, or configuration mismatches
are rejected unless ``--force`` is explicitly supplied.
``fetch-scores --offline`` verifies an existing cache without network access,
while a refresh validates every new artifact before replacing the cache.

Numeric mean imputation, categorical most-frequent imputation, and dense
one-hot encoding are fitted inside each official outer training fold.  Test
rows cannot affect fill values or the category vocabulary.  CTR23's two public
CSV artifacts are pinned to an immutable commit, verified by SHA-256, and
joined by exact numeric OpenML task ID.  The imported values remain
``context_only`` because the MBTR preprocessing and tuning budget do not match
the published learners, even though the outer resampling and mean-fold RMSE do.
Published CTR23 XGBoost performs a 500-evaluation nested random search per
outer fit, while ridge selects its penalty by internal cross-validation;
random forest, the regression tree, and GAM follow different fixed or
model-internal policies.  The reference methods therefore do not share one
tuning budget.

Aggregation proceeds fold → repeat → task → suite.  Every task has equal suite
weight, so the five CTR23 tasks with 100 resamples do not outweigh the thirty
tasks with 10.  Reports use ranks and RMSE ratios across regression datasets;
raw RMSE values in different target units are never averaged.

Running a prepared study
------------------------

A prepared ``.npz`` split contains ``x_train``, ``y_train``, ``x_test``, and
``y_test``, with an optional validation pair.  From a repository checkout:

.. code-block:: bash

   python -m studies.run data/rf1.npz \
     --dataset-id rf1 \
     --protocol-id sketchboost_random_holdout_cv5_2022 \
     --task regression \
     --variant both \
     --run-id 0 \
     --output studies/results/rf1-fold-0.json

The JSON result includes a data hash, dimensions, environment, parameters,
timings, serialized model size, exact metric values, and matching curated
published-reference rows.  A reference row retains its comparison tier; merely
matching a dataset name does not make protocols comparable.

Catalog protocols declare their required fold, repeat, or seed IDs.  Run
``python -m studies.aggregate`` over all per-run JSON files before comparing an
aggregate with published scores; the aggregator rejects missing or duplicate
runs by default.

See the repository's ``studies/README.md`` for dataset preparation rules,
catalog schemas, benchmark limitations, and the complete comparison policy.

Feature-selected linear-response suite
--------------------------------------

The dedicated multivariate manifest contains 16 genuinely multi-output
datasets and 120 primary runs: 90 regression runs across RF1, SCM20D, and eight
MOCR datasets, plus 30 multilabel runs across Mediamill, Delicious, MoA,
Emotions, Scene, and Yeast.

For every run, the benchmark first fits and evaluates ordinary shared MBTR.  It
ranks original input columns by that fitted model's
``feature_importances_``, selects a deterministic positive-gain top set, and
fits the selection's centering and scaling on training rows only.  A second
linear-response model still uses the full feature matrix for tree splits, but
uses the selected standardized columns as the within-leaf design matrix.
Regression uses ridge-linear responses and multilabel classification uses
ridge-linear logit responses.  The classifier defaults to the shared
:math:`1/4` Bernoulli curvature bound; exact per-label Hessian-weighted Grams
are available for smaller designs.  This makes the standard model both the
paired baseline and the screening stage; reports expose incremental linear-fit
time and total two-stage time separately.

The validation-selected protocol caps both fits at 1,000 trees with learning
rate 0.05 and patience 50.  Each fit uses only training rows for parameter
updates, monitors the explicit prepared validation fold, restores its own
validation-best ensemble, and is frozen before the common test set is evaluated
once.  Standard and linear-response MBTR have the same maximum budget but stop
independently, so a paired result can legitimately contain different retained
tree counts.  The intercept-only/base prediction is a candidate as well, so a
fit retains zero trees if no boosting round improves validation loss.  In the
five-fold SketchBoost protocols, each frozen fold model is
evaluated once on the prescribed shared test partition; those test scores are
aggregation replicates and never selection candidates.

Prepared artifacts use
``<protocol-id>/<dataset-id>/<run-id>.npz`` plus a JSON provenance sidecar with
the NPZ checksum.  Twelve datasets and 85 work units are directly downloadable;
MoA needs Kaggle authentication, while the three MEPS files are not bundled by
the pinned MOCR repository.  Install the exact MOCR preparation dependency,
fetch the pinned public sources, and build the prepared subset with:

.. code-block:: bash

   python -m pip install -e '.[studies,mocr-study]'
   python -m studies.multivariate.prepare fetch
   python -m studies.multivariate.prepare prepare

The ``mocr-study`` extra pins NumPy 2.2.2, pandas 2.2.3, and PyTorch 2.6.0.
The fetch phase verifies fixed sizes and SHA-256 values.  The preparation phase
persists feature and target names, array and split identities, source indices,
and preprocessing provenance, and reuses only exact fingerprints.  In
particular, MOCR run IDs 0--9 map to its actual source seeds 2000--2009.

Inspect, execute, and summarize the suite with:

.. code-block:: bash

   python -m studies.multivariate.benchmark plan --data-directory studies/data/multivariate
   python -m studies.multivariate.benchmark run \
     --data-directory studies/data/multivariate \
     --output-directory studies/results/multivariate-es1000-p50 \
     --dataset-id rf1 \
     --dataset-id scm20d \
     --dataset-id mediamill \
     --dataset-id delicious \
     --dataset-id mocr_households \
     --dataset-id mocr_house \
     --dataset-id mocr_bio \
     --dataset-id mocr_blog_data \
     --dataset-id mocr_taxi \
     --n-estimators 1000 \
     --learning-rate 0.05 \
     --early-stopping-rounds 50 \
     --jobs 2 \
     --linear-work-budget 5e11 \
     --linear-time-budget-seconds 1800
   python -m studies.multivariate.benchmark report \
     --output-directory studies/results/multivariate-es1000-p50

These nine directly downloadable datasets provide validation rows and account
for 70 work units.  Add MoA or MEPS dataset filters when their sources have
been prepared.  Use a fresh output directory instead of replacing artifacts
from the earlier fixed 100-tree run.

The larger linear guard is needed because the default 200-billion-work and
300-second estimates would skip RF1 and SCM20D under a 1,000-tree cap; SCM20D
is estimated at roughly 432 billion work and 1,155 seconds.  The guard remains
active and early stopping can finish an actual fit sooner.  Keep the default
guard or run datasets separately when a 30-minute per-stage screening estimate
is too large.  The guard is a cost-model estimate, not a wall-clock timeout.

``--jobs`` parallelizes independent dataset/protocol/run work units, not one
tree fit.  One worker is the conservative default; start with two and increase
only if the largest concurrent fits fit in memory.  Cap BLAS and Numba thread
counts when necessary to avoid oversubscribing the machine, including
``VECLIB_MAXIMUM_THREADS=1`` for Apple Accelerate.  Use one runner as the
coordinator; do not launch separate commands concurrently against the same
output directory because the output lock is not a cross-command mutex.
Progress and completed
trained-model metrics are emitted to stderr, while the final JSON summary
remains on stdout.  Log lines distinguish ``MBTR standard`` from ``MBTR
linear`` during ``run``.  After protocol aggregation and completeness checks,
the ``report`` command emits eligible catalog values using forms such as
``published XGBoost reference rf1: test RMSE=0.020000; tier=direct``.  Such a
published RMSE is provenance context, not a claim that this runner fitted
XGBoost.  Pass ``--log-level WARNING`` to suppress successful-result lines.

The run stage is fingerprinted and independently resumable per variant.  A
bounded-depth work, time, and memory guard can reduce the response-feature
count or skip a prohibitive linear fit without losing the standard result.
Paired summaries include only runs where both variants completed.  Regression
uses RMSE ratios rather than averaging raw errors in unrelated units;
classification uses binary-cross-entropy ratios.  The full data contract,
result layout, guard semantics, and interpretation rules are in
``studies/multivariate/README.md``.

RF1, SCM20D, Mediamill, and Delicious match SketchBoost's pinned split and
headline metric, but its published compute and tuning budget is not the default
MBTR budget.  CatBoost and XGBoost requested up to 30 validation-selected
Optuna trials (TabNet requested 10); CatBoost and XGBoost then used long
early-stopped refits.  A 32-hour timeout made realized candidate counts dataset
dependent.  SketchBoost inherited the selected CatBoost structure, and its
reduced-dimension summary selected the dimension by mean test loss.  The Top
Outputs, Random Sampling, and Random Projection rows are therefore retained
only as ``context_only`` archival records and excluded from clean comparison
tables, headline selection, and rankings.  Full-output SketchBoost and the
published baselines remain eligible as direct data/split/metric references,
although they are not an unbiased budget-matched leaderboard until MBTR is
tuned under a shared validation-only search protocol.

The implemented Emotions, Scene, and Yeast runs are fresh comparisons on the
KDIS five-fold archives with a fixed threshold of 0.5.  They are not directly
comparable to the historical Mulan holdout scores or UCO 2024's ten-partition
results.  These KDIS artifacts have no validation pair.  The runner therefore
rejects them when early stopping is enabled rather than using the outer test
fold for selection.  ``--disable-early-stopping`` permits an explicitly
labelled fixed-1,000-tree diagnostic only; a clean early-stopped KDIS study
requires a reproducible inner split in each outer training fold and a distinct
nested-validation protocol identifier.
