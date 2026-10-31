# Multivariate MBTR benchmark

This study runs the repository's genuinely multi-output datasets under one
explicit manifest. It compares ordinary shared-tree MBTR with a two-stage
linear-response MBTR on both regression and multilabel classification tasks.

The primary manifest contains 16 datasets and 120 runs:

| Task | Datasets | Runs |
|---|---|---:|
| Multi-output regression | RF1 and SCM20D (5 each); Households, House, Bio, Blog Data, Taxi, MEPS19, MEPS20, and MEPS21 (10 each) | 90 |
| Multilabel classification | Mediamill, Delicious, MoA, Emotions, Scene, and Yeast (5 each) | 30 |

The protocol-to-dataset mapping and expected run IDs are fixed in
[`manifest.json`](manifest.json) and cross-checked against the main study
catalog before execution. The runner does not download source data implicitly.

Twelve datasets (85 work units) are directly downloadable without an account:

| Source | Direct datasets | Runs |
|---|---|---:|
| SketchBoost/OpenML/XC | RF1, SCM20D, Mediamill, Delicious | 20 |
| Pinned MOCR repository | Households, House, Bio, Blog Data, Taxi | 50 |
| Fixed KDIS archives | Emotions, Scene, Yeast | 15 |

MoA requires Kaggle authentication and acceptance of its competition terms.
MEPS19/20/21 are referenced by MOCR but are not bundled in its pinned public
repository. The public preparation command reports these as unavailable rather
than replacing them or silently changing their protocols.

## Linear-response protocol

Every run uses a leakage-safe, two-stage procedure:

1. Fit an ordinary shared-tree `MBTRegressor` or `MBTClassifier`, according to
   the manifest task, on the training rows while monitoring the prepared
   validation rows.
2. Retain the validation-best ensemble, freeze it, and evaluate it once on the
   common test set as the standard baseline.
3. Rank original input columns by that frozen model's training-derived
   `feature_importances_` (summed split gain), using the original column index
   to break ties.
4. Select the smallest positive-importance top set that reaches the configured
   importance coverage, subject to the feature cap and compute guard.
5. Fit the selected columns' mean and standard deviation on training rows only.
6. Fit a new linear-response estimator. Regression uses
   `MBTRegressor(objective="linear_regression")`; classification uses
   `MBTClassifier` with linear-logistic leaves. All original columns remain
   available for tree splits, while only the selected, train-standardized
   columns form the within-leaf response design matrix. This second model
   monitors the same prepared validation fold and retains its own best
   iteration; it is not forced to use the standard model's tree count.
7. Freeze the validation-best linear-response model, apply the unchanged
   feature transform, and evaluate the common test set once.

Thus the standard score is not a disposable screening fit: it is the paired
baseline, and its fit time is included in the linear procedure's end-to-end
cost. The linear ridge penalty is configured separately because it regularizes
leaf coefficients and is not equivalent to ordinary leaf regularization.
All CLI settings are treated as fixed study configuration. If they are tuned,
that tuning must use only an inner split of the outer training data or the
declared validation partition—never the benchmark test targets.

The default study budget is a cap of 1,000 trees at learning rate `0.05`, with
patience 50 on an explicit prepared validation fold. The cap is not the
reported model size: each variant is restored to its own validation-best
iteration. The intercept-only/base prediction is also a candidate, so a model
retains zero trees if no boost improves validation loss. Test predictions and
metrics are computed only after that choice is fixed. In the five-fold
SketchBoost protocols this means each frozen fold model
is evaluated once on the same prescribed test partition; the five test scores
are aggregation replicates, not candidates for selecting a tree count or any
other setting.

For classification, a constant-response leaf stores one constant logit update
per label. A linear-response leaf instead stores a ridge-regularized linear
coefficient vector for each label's logit update and evaluates it on the
selected design columns. Both
models optimize Bernoulli cross-entropy; no classification run is converted to
squared-error regression. The default
`--linear-response-curvature upper_bound` uses the global Bernoulli curvature
bound `1/4` and shares one design Gram matrix across labels. `exact` uses a
probability-weighted Gram matrix per label. It can be useful for small problems,
but its temporary memory and matrix work grow with the label count and the
square of the response-design width.

If the standard model creates no positive-gain split, no arbitrary response
column is chosen. The linear stage is recorded as skipped. A bounded-depth
work, time, and temporary-memory estimate can reduce the selected top-*k* or
skip a computationally prohibitive fit. `--force-linear` is an explicit escape
hatch and its use is recorded in the result.

## Prepared-data layout

Install the study dependencies before preparing the complete public subset:

```bash
python -m pip install -e '.[studies,mocr-study]'
```

`numpy==2.2.2`, `pandas==2.2.3`, and `torch==2.6.0` are deliberately pinned by
`mocr-study`: these match the upstream data path, seeded `random_split`, and
float32 reduction behavior. Fetch and checksum all 12 public sources, then
generate all 85 runs with:

```bash
python -m studies.multivariate.prepare fetch
python -m studies.multivariate.prepare prepare
```

Or perform both phases with `python -m studies.multivariate.prepare all`. Use
`--dataset-id emotions` (repeatable) for a bounded preparation. Existing exact
artifacts are verified and reused; a mismatched raw or prepared artifact fails
unless `--replace` is explicitly supplied. Raw files default to
`studies/data/multivariate-raw/` and prepared files to
`studies/data/multivariate/`; both stay outside the distributed package.

The adapters pin source sizes and SHA-256 values and retain array hashes,
feature/target names, preprocessing, split identities, and source row indices
where they exist. Important source-specific details are reproduced explicitly:

- RF1/SCM20D use SketchBoost's seed-42 random 80/20 holdout and five shuffled
  training folds. RF1's missing inputs receive training-fold mean imputation
  because MBTR requires finite values; this difference from native missing-value
  learners is recorded in every sidecar.
- Mediamill/Delicious use the pinned SketchBoost Extreme Classification
  Repository ZIPs and the first of their ten predefined split-index columns,
  not an inferred Mulan row order.
- MOCR run IDs 0–9 map to upstream source seeds 2000–2009. The pinned
  preprocessing actually produces 14, 17, 8, 269, and 4 features for
  Households, House, Bio, Blog Data, and Taxi, respectively. Persisted source
  indices are zero-based positional CSV data rows before preprocessing, not
  pandas index labels or positions inside the seed-specific subsample.
- KDIS label XML is treated as a set while target order follows ARFF declaration
  order. The fixed 2017 iterative-stratified archives are identified by hash;
  KDIS does not state dataset license terms or the MLDA generator seed/version.
  These archives contain only outer train/test folds, not validation rows.
  Consequently they are outside the validation-early-stopped protocol above:
  using early stopping or tuning on KDIS requires a reproducible inner split
  inside every outer training fold and defines a separate nested protocol.

Place every prepared run below one data directory:

```text
prepared/
  <protocol-id>/
    <dataset-id>/
      <run-id>.npz
      <run-id>.json
```

The numeric `.npz` contract is the same as the generic study runner:

- required `x_train`, `y_train`, `x_test`, and `y_test` arrays;
- optional `x_validation` and `y_validation`, provided together;
- optional `target_offset` and positive `target_scale`, provided together when
  regression targets were standardized using training rows;
- optional calibration arrays may be retained for MOCR provenance, but this
  point-prediction benchmark never uses them.

The JSON sidecar must identify `dataset_id`, `protocol_id`, and `run_id`, and
contain `split_sha256`, the SHA-256 of the adjacent NPZ. It should also retain
source URLs and revisions, raw-file checksums, license or access terms, split
index hashes, preprocessing details, and `feature_names`. Strings belong in
the sidecar because NPZ files are opened with pickles disabled. Imputation,
encoding, feature transforms, and target transforms must be fitted on outer
training rows only.

## Running the suite

Install the optional external boosters before requesting the local XGBoost,
LightGBM, and CatBoost comparison:

```bash
uv pip install --python .venv/bin/python -e '.[studies,external-boosters]'
```

First inspect the full resolved plan and missing artifacts:

```bash
python -m studies.multivariate.benchmark plan \
  --data-directory studies/data/multivariate
```

Run the nine directly downloadable datasets that already provide validation
rows (70 work units) and write independently resumable results:

```bash
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
  --max-depth 4 \
  --linear-max-features 8 \
  --linear-work-budget 5e11 \
  --linear-time-budget-seconds 1800 \
  --linear-response-curvature upper_bound
```

Run the same validation-selected protocol for every local model with one
native thread and one work-unit process for interpretable timing:

```bash
caffeinate -dimsu env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 \
  .venv/bin/python -m studies.multivariate.benchmark run \
  --data-directory studies/data/multivariate \
  --output-directory studies/results/multivariate-all-models-es1000-p50 \
  --dataset-id rf1 \
  --dataset-id scm20d \
  --dataset-id mediamill \
  --dataset-id delicious \
  --dataset-id mocr_households \
  --dataset-id mocr_house \
  --dataset-id mocr_bio \
  --dataset-id mocr_blog_data \
  --dataset-id mocr_taxi \
  --models-all \
  --native-threads 1 \
  --jobs 1 \
  --n-estimators 1000 \
  --learning-rate 0.05 \
  --early-stopping-rounds 50 \
  --linear-work-budget 5e11 \
  --linear-time-budget-seconds 1800 \
  --log-level INFO
```

This all-model comparison includes XGBoost's experimental shared-output-tree
mode as a separately labelled row. Use repeated `--model` options instead of
`--models-all` to request a subset. A completed external stage is independently
resumable, so adding a method later does not retrain completed stages.

Add `--dataset-id moa` or the three MEPS IDs when those sources have been
prepared. The KDIS Emotions, Scene, and Yeast artifacts cannot be added to this
command because they lack validation rows. When early stopping is enabled the
runner rejects such a work unit instead of leaking its outer test fold into
model selection. `--disable-early-stopping` is available for an explicitly
labelled fixed-1,000-tree diagnostic, but that is not the validation-selected
protocol and is not a substitute for creating nested KDIS splits.

To request all validation-capable manifest IDs while tolerating unavailable
MoA/MEPS sources, repeat their `--dataset-id` values and pass
`--allow-missing`. Those omissions remain visible in `run-summary.json`; a
partial suite is never treated as complete. Do not use an unfiltered
early-stopping run until KDIS has a separately identified nested-validation
protocol.

The enlarged linear guard in this command is intentional: under the
1,000-tree cap, the default `2e11` work and 300-second estimates would skip the
RF1/SCM20D linear stages (SCM20D is estimated at about `4.32e11` work and 1,155
seconds). The guard still applies to every run, and an actual fit may finish
earlier because early stopping retains fewer than 1,000 trees. Use the default
guard for a conservative run, or target datasets separately, if 30 minutes per
linear stage is not an acceptable screening estimate. This guard is a cost-model
estimate, not a wall-clock timeout.

`--jobs` parallelizes independent dataset/protocol/run work units; it does not
split a single tree fit. One worker remains the default for conservative memory
use. Start with `--jobs 2`, watch resident memory, and scale up only when the
largest concurrent fits fit comfortably. When numerical libraries use internal
threads, cap them to avoid multiplying worker and BLAS parallelism, for example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -m studies.multivariate.benchmark run ... --jobs 2
```

Use one benchmark command as the process coordinator. Do not launch separate
commands concurrently against the same output directory; `--jobs` provides the
supported parallelism and the output lock is not a cross-command mutex.

Experimental top-gradient output sketching is available through
`--output-sketch-size K`. It reduces only the split-search output dimension;
each chosen leaf is recomputed with all outputs and full-output validation loss
still selects the retained iteration. Use a separate output directory for each
predeclared `K`, and choose among values using validation results only. The
RF1/SCM20D speed gate did not justify sketching their 8/16 outputs; the option
is intended primarily for high-output datasets such as Mediamill and
Delicious.

Progress records are emitted immediately to stderr, so redirecting stdout still
produces valid JSON. During `run`, a completed line identifies the dataset, run,
trained MBTR variant, headline test metric, retained/evaluated tree counts, and
fit time, for example `completed rf1/0 MBTR linear: test RMSE=0.010000
retained=19 evaluated=69 fit=12.3s`. The `report` command emits eligible
published-reference lines only after protocol aggregation and completeness
checks. Catalog values are explicitly labelled `published ... reference`; a
line such as `published XGBoost reference rf1: test RMSE=0.020000; tier=direct`
does not mean XGBoost was fitted by this runner. Reused, guarded, skipped, and
failed stages are also labelled.
Use `--log-level WARNING` to suppress successful-result lines.

Use `--dataset-id rf1` (repeatable) for a bounded run. By default missing
prepared files or sidecars are errors; `--allow-missing` and
`--allow-missing-sidecars` are intended for explicitly labelled development
runs. Use a fresh output directory for this protocol rather than replacing
artifacts from the earlier 100-tree fixed-budget run. Exact split, source, and
configuration fingerprints resume without
refitting. A failed or guarded linear stage does not discard its completed
standard result. A mismatched lock fails conservatively; use `--replace` only
after reviewing the changed configuration. Standard artifacts remain reusable
when only linear-selection, linear-ridge, or linear-guard settings change.

Generate JSON, CSV, and Markdown summaries with:

```bash
python -m studies.multivariate.benchmark report \
  --output-directory studies/results/multivariate-es1000-p50
```

Regenerate the two root-README figures from a completed aggregate with:

```bash
MPLBACKEND=Agg python -m studies.multivariate.plot_results \
  --aggregate studies/results/multivariate-all-models-mbtr-refresh-20261031-es1000-p50/aggregate.json \
  --output-directory docs/source/_static
```

This command requires the optional plotting dependencies (`.[plot]`). It keeps
the README's dated eight-dataset snapshot pinned even if later runs complete;
the figures show only the four external methods relative to standard MBTR.
Repeat `--dataset-id ID` in the desired order to generate a different complete-
dataset selection, and update the README's dated interpretation at the same
time.

The dated directory above is a 31 October 2026 refresh of only the standard and
linear-response MBTR stages after the implementation speedups. It retains the
four external-model stages from the preceding same-split, one-thread run so the
quality comparison stays paired. External/MBTR timing ratios therefore compare
separately recorded local runs rather than contemporaneous wall-clock fits.
Linear-response artifacts remain in the aggregate and full report, but are
intentionally excluded from these external-comparison figures because their
feature-selection and second-fit protocol is not a like-for-like single-model
baseline.

Per-run artifacts live at:

```text
results/<dataset-id>/<run-id>/standard.json
results/<dataset-id>/<run-id>/linear_response.json
```

Each standard result stores the complete split-gain vector and stable ranking.
Each completed linear result stores selected indices and names, covered gain,
the training-only scaling parameters, cost estimate, incremental fit time, and
two-stage total time. All variants include predictive metrics,
prediction time, serialized model size, resolved backend, and an
environment/source fingerprint, as well as the retained iteration and
validation-loss history.
When a prepared regression split includes `target_offset` and
`target_scale`, the artifacts retain both stored-scale metrics and metrics after
inverse transformation to raw target units; paired headline comparisons use
the raw-unit metrics.

## Interpreting results

Comparisons use only run IDs for which both variants completed. Regression
summaries use paired `global_rmse` deltas and ratios; classification summaries
use paired binary-cross-entropy deltas and ratios, with lower values preferred
for both tasks. Reports also include win/tie/loss counts, selection frequency,
and incremental versus end-to-end time. They do not average raw RMSE across
datasets with different units; suite summaries use equal-dataset normalized
ratios. Published values appear only for a complete matching catalog protocol
and retain their declared comparison tier. A guarded fit reports what budget
excluded it and is not treated as a paired accuracy result.

The standard and linear-response entries in a pair may legitimately retain
different numbers of trees: both receive the same 1,000-tree maximum,
learning rate, patience, and validation data, but early stopping is independent
because they optimize different leaf models. Compare their frozen test metrics,
not an arbitrarily equal tree count. For MOCR, early stopping monitors the
stored, train-standardized target objective; raw-unit RMSE is computed only
after inverse transformation and remains the headline paired metric.

### Comparability with published SketchBoost scores

RF1, SCM20D, Mediamill, and Delicious match the pinned SketchBoost data split
and headline metric, but the default MBTR command is intentionally an untuned
fixed configuration. The published comparison first evaluated a default, then
requested up to 30 Optuna trials for CatBoost/XGBoost (10 for TabNet), selecting
parameters by mean validation loss. CatBoost/XGBoost were then refit with
learning rate 0.015, up to 20,000 trees, and validation-based early stopping.
(That refit does not describe TabNet.) A 32-hour Optuna timeout made the
realized search budget uneven. Complete-candidate counts, including the
default, were CatBoost=31, XGBoost=31, and TabNet=11 for both RF1 and SCM20D;
Delicious used 5/10/11 and Mediamill 8/31/11, respectively. MoA, when
available, used 10/31/10. SketchBoost inherited CatBoost's selected structural
parameters and used the same lower-learning-rate refit. Its summary tables
selected the reduced sketch dimension by mean **test** loss, so those strategy
headline values have test-selection optimism. Consequently, the Top Outputs,
Random Sampling, and Random Projection rows are catalogued as `context_only`:
reports may retain them in a clearly marked archival section, but clean
comparison tables, headline selection, and rankings must exclude them.
Full-output SketchBoost and the baseline rows do not have that
dimension-selection step and remain eligible as direct data/split/metric
references.

The remaining direct rows are still not a budget-matched hyperparameter
contest. A tuned MBTR comparison must define one shared search space and budget,
select every parameter—including linear ridge, importance coverage, and
response width—using only the declared validation partitions, freeze that
configuration, refit/retain each validation-best model, and then evaluate each
fitted fold model once on the shared test set without using those test scores
for selection. The practical MBTR schedule documented here (`0.05`, at most
1,000 trees, patience 50) is fixed in advance; it does not claim compute-budget
parity with SketchBoost's `0.015`, at most 20,000-tree refit.

For RF1, “direct” means identical raw rows, targets, split, and metric—not
identical missing-value treatment. Published PyBoost/CatBoost/XGBoost retained
the 3,264 missing inputs and handled them natively, while finite-input MBTR uses
training-fold mean imputation (the same leakage-safe policy used by the
published TabNet path).

The historical KBMF scores for Emotions/Scene/Yeast use a different fixed Mulan
holdout and are context only for the KDIS five-fold runs. The UCO 2024
comparison is also distinct: its official data bundle contains ten
train/generalization partitions, not these five KDIS folds. No open score is
currently treated as direct for this KDIS protocol. Because the prepared KDIS
folds have no validation partition, they must not silently fall back to using
the outer test fold for early stopping; a nested inner-validation study should
receive a distinct protocol identifier. MOCR publishes conformal-
region results rather than a directly comparable point-prediction leaderboard,
so this study creates new RMSE baselines on its persisted splits.

Feature importance is a model property, not proof of causal relevance. The
selection experiment tests whether those split-relevant inputs improve the
within-leaf response surface. It should be interpreted together with selection
stability across folds or seeds and the full two-stage compute cost.
