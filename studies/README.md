# MBTR scientific studies

This directory is for accuracy, quality, and external-baseline studies. It is
separate from `benchmarks/`, whose ASV jobs measure implementation speed and
memory on controlled synthetic matrices. Study data and generated results are
not part of the Python package or wheel.

## What belongs here

The evidence is deliberately layered:

| Layer | Suites | What it can establish |
|---|---|---|
| Primary multi-output | SketchBoost RF1, SCM20D, Mediamill, Delicious, and MoA | Accuracy and cost of a shared partition versus independent outputs and published multi-output methods |
| Historical continuity | Selected Mulan MTR/MLC protocols | Compatibility with older literature and small offline smoke studies |
| Modern scalar breadth | TALENT-tiny, then TALENT basic | Whether MBTR is a competitive general scalar tabular learner across domains |
| Standardized scalar classification | Binary subset of OpenML-CC18 | Reproducible OpenML tasks, fixed folds, and shareable runs |

The last two layers **cannot** demonstrate a multivariate advantage: every
compatible task has one target. On such a task, shared and independent MBTR
are the same model. They are still useful for detecting weaknesses in
preprocessing, calibration, ordinary regression/classification quality, and
runtime scaling.

Mulan is dated. It remains in the study only because its datasets provide
continuity with published multi-target and multilabel work. It should not be
the sole or headline benchmark.

### More modern genuinely multi-output studies

For multilabel classification, the strongest expansion found is the
[UCO 2024 comparison](https://cib.uco.es/comp-class-perf/) paired with the
[KDIS fixed partitions](https://www.uco.es/kdis/mllresources/). It evaluates 62
methods (197 configurations) over 65 datasets and publishes detailed results.
Start with a dense subset such as Birds, CAL500, Emotions, Flags, Scene,
Water-quality, and Yeast using the iterative-stratified five-fold partitions.
Before calling UCO values direct comparisons, verify that the downloadable
result bundle used those identical folds; otherwise classify them as protocol
references.

For regression, there is no equally modern public point-prediction leaderboard.
The best data expansion found is the
[ICML 2025 multi-output conformal-regression corpus](https://proceedings.mlr.press/v267/dheur25a.html),
which provides 13 tabular multi-output datasets and a seeded split algorithm.
Its published metrics concern conformal coverage and region size, not point
RMSE. We therefore reproduce the splitter from pinned commit
`6d40a669492ef1c7860568315796a99a895d9548`, persist the generated indices,
and run fresh shared MBTR,
independent MBTR, CatBoost, per-target XGBoost, Random Forest/Extra Trees,
regressor-chain, and constant baselines. `catalog/datasets.json` already lists
the principal modern additions and `mocr_point_prediction_2025` records the
adapted point-prediction protocol. The upstream pipeline standardizes both
features and targets on training rows; report standardized-target results and
raw-unit metrics from inverse-transformed predictions as separate scales.

## Current benchmark assessment

### SketchBoost

This is the strongest public comparison source found so far. It contains two
multi-output regression and three multilabel tasks, publishes exact means and
standard deviations, and provides evaluation code. The original protocol has
an important limitation: models from five training folds are evaluated on one
common test set, and the reduced sketch dimension was selected by the final
test loss. That makes the Top Outputs, Random Sampling, and Random Projection
headline values unsuitable for an unbiased comparison or ranking. They remain
in the catalog as `context_only` archival records, explicitly labelled
`test-selected`, and are excluded from clean tables and rankings. Full-output
SketchBoost and the published baselines do not use that dimension-selection
step and remain eligible as direct data/split/metric references. A separate,
nested validation protocol is required to compare reduced sketch dimensions.

The exact public summary values are transcribed in
`references/published_scores.csv`, together with their metric, protocol, source,
scale, and comparison tier. Keeping the excluded rows provides provenance
without allowing them to influence a clean headline result.

### OpenML-CC18 and other OpenML suites

[OpenML benchmark suites](https://docs.openml.org/benchmark/) are valuable
because a task fixes the dataset, target, and train/test folds, and public runs
can be downloaded and reused. OpenML-CC18 is suite 99 and contains 72
**single-target classification** tasks: 35 binary and 37 multiclass. It is not a
regression collection. Current `MBTClassifier` can enter only its binary
subset. The multiclass tasks require a categorical multiclass objective and
must not be emulated by calling the multilabel interface on one-hot targets.

OpenML-CTR23 (suite 353) provides 35 single-target regression tasks and exact
published RMSE results for XGBoost, random forest, GAM, ridge, and a regression
tree. Its live task splits disagree with the paper prose: the task definitions
use cross-validation, including five repeated-CV tasks, rather than the stated
large-dataset holdouts. Reproduction must use and pin the downloaded task split
indices. `openml/manifest.json` records the audited suite identities and this
discrepancy.

The runnable OpenML pipeline implements that policy. Install it with
`python -m pip install -e '.[studies]'`, then run a bounded end-to-end check:

```bash
python -m studies.openml.pipeline all \
  --suite ctr23 \
  --workspace studies/data/openml-ctr23-smoke \
  --max-tasks 1 \
  --max-runs-per-task 2 \
  --n-estimators 10 \
  --engine numpy
```

For a complete study, omit both limits. The equivalent staged commands are
`prepare`, `run`, `aggregate`, `fetch-scores`, and `report`; see
[`openml/README.md`](openml/README.md) for their exact invocations and workspace
layout. Exact matching artifacts resume, while fingerprint mismatches fail
unless replacement is explicitly requested with `--force`. Score downloads
are explicit and checksum-verified; `fetch-scores --offline` validates an
existing cache without making a network request.

The preparer persists OpenML's official positional indices and fits numeric
mean imputation, categorical most-frequent imputation, and dense one-hot
encoding on each outer training fold only. CC18 multiclass tasks are recorded
as unsupported rather than reinterpreted as multilabel tasks. CTR23 scores are
pinned at an immutable source revision and joined by exact numeric task ID, but
are labelled `context_only`: the outer resampling and fold-RMSE aggregation
align, while preprocessing and tuning do not. Published CTR23 XGBoost uses a
500-evaluation nested random search per outer fit, and ridge selects its penalty
by internal cross-validation; the other learners use different fixed or
model-internal fitting policies. There is no common published tuning budget.

Aggregation is fold → repeat → task → suite. Every dataset has equal suite
weight, including when one CTR23 task exposes 100 resamples and another exposes
10. Reports compare heterogeneous regression tasks using ranks and RMSE ratios,
not an average of raw, differently scaled RMSE values.

Other OpenML regression suites are also single-target.
Public OpenML runs are useful context, not automatically a controlled baseline:
flows can differ in preprocessing, tuning budget, versions, and compute. A
comparison should either rerun selected methods under one protocol or filter
public runs to a fully specified study.

### TALENT / LAMDA

[TALENT](https://github.com/LAMDA-Tabular/TALENT) is the better modern breadth
study. Its basic benchmark contains 300 scalar tasks: 120 binary, 80
multiclass, and 100 regression. TALENT-tiny contains 45 selected tasks: 15
binary, 12 multiclass, and 18 regression. MBTR therefore supports 220/300
basic tasks and 33/45 tiny tasks today.

Start with those 33 TALENT-tiny tasks. The published protocol uses fixed
64%/16%/20% train/validation/test partitions, 100 Optuna trials per
method-dataset pair, and 15 seeds. Classification is primarily ranked by
accuracy and regression by RMSE. Public per-dataset result tables are available
for [binary classification](https://github.com/LAMDA-Tabular/TALENT/blob/main/results/cls_bin.md)
and [regression](https://github.com/LAMDA-Tabular/TALENT/blob/main/results/regression.md),
including means and standard deviations for classical and neural baselines.

For a publishable comparison, pin all of the following:

- TALENT Git commit (the project currently has no tagged releases);
- dataset archive SHA-256 and selected-dataset manifest;
- source-level dataset terms, which are not inherited from TALENT's MIT code
  license;
- preprocessing fitted on training data only;
- tuning metric, budget, and seed list.

The current TALENT repository has evolved beyond the paper protocol, including
binary decision-threshold tuning. Published-score comparisons must follow the
paper's pinned protocol rather than whatever `main` does at execution time.
Across heterogeneous regression datasets, aggregate ranks or normalized errors;
do not average raw RMSE values.

## Catalog layout

- `catalog/datasets.json`: dimensions, source, representation, and caveats.
- `catalog/protocols.json`: immutable split, selection, metric, and aggregation
  definitions.
- `catalog/suites.json`: suite scope, current MBTR coverage, and evidential role.
- `references/published_scores.csv`: values transcribed from public tables.
- `metrics.py`: executable metric definitions.
- `adapters.py`: shared-tree and one-model-per-output comparison adapters.
- `run.py`: an offline runner over prepared NumPy splits.
- `fetch_scores.py`: immutable, checksum-verified public-score retrieval.
- `openml/pipeline.py`: staged OpenML preparation, execution, aggregation, and
  report generation.
- `multivariate/benchmark.py`: the 16-dataset, 120-run shared MBTR versus
  feature-selected linear-response study.

Validate all cross-references and score metadata with:

```bash
python -c "from studies.catalog import validate_catalog; validate_catalog()"
```

## Prepared split contract

The generic runner never downloads a dataset implicitly. Prepare and archive an
`.npz` file yourself, or use an explicit suite-specific preparation stage such
as `studies.openml.pipeline prepare`. The split contract contains:

- required `x_train`, `y_train`, `x_test`, and `y_test` arrays;
- optional `x_validation` and `y_validation` arrays, provided together;
- optional one-dimensional `target_offset` and positive `target_scale` arrays,
  provided together when stored regression targets were train-standardized;
- two-dimensional finite numeric feature matrices;
- a one- or two-dimensional target array;
- binary `0`/`1` targets for binary and multilabel studies.

When target-transform arrays are present, the model fits the stored standardized
targets but the runner inverse-transforms training targets, test targets, and
predictions before reporting raw-unit metrics. The transform must be fitted on
training rows only. Run a separate explicitly labelled standardized-scale study
when that metric is also scientifically relevant, passing
`--stored-target-scale train_standardized` when no inverse transform is stored.

Record the downloader, raw-file checksums, license/terms, split construction,
and preprocessing code next to the prepared artifact. Never fit imputers,
encoders, scalers, or target transforms on validation or test rows.

Run a controlled shared-versus-independent comparison with:

```bash
python -m studies.run data/rf1.npz \
  --dataset-id rf1 \
  --protocol-id sketchboost_random_holdout_cv5_2022 \
  --task regression \
  --variant both \
  --run-id 0 \
  --n-estimators 500 \
  --output studies/results/rf1-fold-0.json
```

The result records the split SHA-256, dimensions, estimator parameters, Python,
NumPy and MBTR versions, fit/predict wall time, serialized model size, model
count, metrics, and any matching rows from the curated published-score table.
These rows retain their comparison tier and protocol notes; their presence is
not a claim of direct comparability. Timing is descriptive scientific-study
metadata; use ASV in `benchmarks/` for stable performance regression tests.
For speed claims, run each variant in a separate process and counterbalance cold
and warm execution order; a combined `--variant both` run is for convenience,
not a hardware-fair timing comparison.

For the primary genuinely multi-output suite, use the dedicated
[`multivariate` benchmark](multivariate/README.md). It fits standard MBTR first,
uses that training model's split-gain importances to select original input
columns, and supplies only those train-standardized columns as the design
matrix for linear-response leaves. The manifest covers 90 regression runs and
30 multilabel runs. Regression leaves fit ridge-linear responses;
classification leaves fit ridge-linear logits under the same Bernoulli loss as
the standard classifier. The original full feature matrix remains available
for tree routing in both cases.

The validation-selected multivariate protocol caps both variants at 1,000
trees with learning rate `0.05` and patience 50. Each model is fitted on the
training rows, monitored on the prepared validation fold, restored to its own
validation-best iteration, frozen, and only then evaluated once on the common
test set. Standard and linear-response MBTR therefore share a maximum budget
but may retain different tree counts. The KDIS archives contain only outer
train/test folds; they require a reproducible inner split and a separately
identified nested protocol before early stopping or tuning can be used. The
outer KDIS test fold must never be reused as validation.

The multivariate runner accepts `--jobs N` to execute independent dataset/run
work units concurrently. Start with two workers and increase only when memory
allows; each fit may also use threaded numerical kernels, so unrestricted BLAS
threads can oversubscribe the machine; on macOS the relevant cap includes
`VECLIB_MAXIMUM_THREADS=1`. Live trained-model metrics stream to stderr during
`run`, leaving the final JSON summary on stdout. After aggregation and
completeness checks, `report` logs eligible literature values as published
references, never as locally fitted models.

Multi-run protocols are not publishable from one JSON file. After producing all
required fold/repeat/seed files, validate completeness and aggregate them:

```bash
python -m studies.aggregate studies/results/rf1-fold-*.json \
  --output studies/results/rf1-aggregate.json
```

The command rejects duplicate, unexpected, or missing run IDs; checks model and
metric configuration; computes mean and sample standard deviation; and exposes
published references only when the catalog protocol is complete. Use
`--allow-incomplete` only for diagnostic summaries.

## Comparison rules

1. Compare a published value only when dataset version, split, preprocessing,
   selection budget, metric definition, and aggregation all match.
2. Treat `direct`, `protocol_reference`, and `context_only` catalog tiers
   differently. Do not silently promote a historical or rounded value.
3. Report shared and independent MBTR with the same hyperparameter budget. Also
   report model count, serialized size, fit time, and prediction time.
4. Keep test data out of model, iteration, feature, and hyperparameter
   selection. Select the validation-best iteration, freeze the configuration,
   and evaluate each resulting model on its test set once. If reproducing a
   test-selected paper protocol, label it and add a clean nested-validation run;
   exclude such rows from clean tables and rankings.
5. Preserve fold-level measurements. In particular, average fold aRRMSE rather
   than pooling predictions across folds.
6. Do not use the old literature's elementwise multilabel "accuracy" without
   naming it. Report BCE, Hamming loss, subset accuracy, micro-F1, and macro-F1
   explicitly.
7. Do not use OpenML or TALENT scalar breadth as evidence that shared response
   structure helps. That claim requires genuinely multi-output targets.

Generated data and result files should stay local or be published as versioned
release artifacts; only small curated manifests and cited reference values
belong in Git.
