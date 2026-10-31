# OpenML scalar generalization studies

OpenML is a secondary study, not a replacement for the genuinely multi-output
suites. Current OpenML supervised tasks have one target feature, and the Python
client does not split multiple target columns. Do not group unrelated targets
or one-hot a multiclass target and call the result a multivariate benchmark.

The pinned suite identities and audited task counts are in `manifest.json`:

- CC18 (`99`): 35 supported binary and 37 currently unsupported multiclass
  classification tasks;
- CTR23 (`353`): 35 supported scalar regression tasks;
- AMLB classification (`271`): 41 supported binary and 30 unsupported
  multiclass tasks;
- AMLB regression (`269`): 33 supported scalar regression tasks.

Start with CC18 binary and CTR23. AMLB is a later large-scale study, not a CI
target. The unversioned OpenML collections `454` and `455` were still marked
`in_preparation` when audited on 2026-10-28 and are intentionally excluded.

## Runnable pipeline

Install the optional study dependencies from a repository checkout:

```bash
python -m pip install -e '.[studies]'
```

The pipeline has explicit, independently resumable stages:

```bash
python -m studies.openml.pipeline prepare --suite ctr23 --workspace studies/data/openml-ctr23
python -m studies.openml.pipeline run --workspace studies/data/openml-ctr23
python -m studies.openml.pipeline aggregate --workspace studies/data/openml-ctr23
python -m studies.openml.pipeline fetch-scores --workspace studies/data/openml-ctr23
python -m studies.openml.pipeline report --workspace studies/data/openml-ctr23
```

`all` runs the same stages in order. This bounded command is useful for checking
the installation before starting the complete 800-resample CTR23 workload:

```bash
python -m studies.openml.pipeline all \
  --suite ctr23 \
  --workspace studies/data/openml-ctr23-smoke \
  --max-tasks 1 \
  --max-runs-per-task 2 \
  --n-estimators 10 \
  --engine numpy
```

That is a smoke test, not a published-score comparison: the report records that
the official task resampling is incomplete. Use `--task-id` to select exact
tasks, or omit all selection limits for the resolved suite workload. CC18
preparation automatically records multiclass tasks as unsupported; it runs only
the 35 binary tasks because the current classifier has no categorical
multiclass objective.

The workspace is a reproducibility artifact:

| Path | Contents |
|---|---|
| `suite-lock.json` | exact suite task list, dataset/task metadata, selection, preprocessing policy, and failures |
| `run-lock.json` | model configuration and source-tree fingerprint for the active run |
| `openml-cache/` | OpenML client cache |
| `indices/task-*/` | unmodified official positional train/test indices |
| `prepared/task-*/` | finite numeric `.npz` folds and provenance sidecars |
| `results/task-*/` | one MBTR result per official resample |
| `aggregates/` | completeness-checked task aggregates |
| `references/` | checksum-verified source files and normalized public scores |
| `summary.json`, `summary.csv`, `summary.md` | machine- and human-readable suite reports |

Preparation and fitting reuse an existing artifact only when its fingerprint
and checksum match. Fingerprints include the relevant preparation/model source
files, so uncommitted implementation changes cannot silently reuse stale
folds or predictions. A changed selection, preprocessing policy, prepared
split, model configuration, or source tree is rejected instead of being mixed
into a study. Use a new workspace for a new experiment; `--force` is available
only for an intentional replacement. Failures are persisted so successful
tasks do not need to be repeated.

Score retrieval is an explicit network action. After one verified download,
the cache can be checked without network access:

```bash
python -m studies.openml.pipeline fetch-scores \
  --workspace studies/data/openml-ctr23 \
  --offline
```

`--offline` fails on a missing, incomplete, or checksum-invalid cache.
`--refresh` downloads both immutable artifacts again, validates them completely,
and only then replaces the valid cache.

## Required preparation behavior

For each run:

1. Resolve the suite to exact task IDs and save that list with retrieval time.
2. Download and save OpenML's train/test indices; do not regenerate the folds.
3. Preserve dataset ID/version, task ID, target name, license, citation, and a
   canonical hash of the selected public metadata fields. OpenML's data MD5 is
   recorded separately and is not mislabelled as a raw metadata hash.
4. Fit numeric mean imputation and most-frequent categorical imputation plus
   dense one-hot encoding inside each official training fold. Unseen categories
   are ignored (or mapped to the recorded rare-level bucket when cardinality is
   capped); test rows never influence the vocabulary or fill values.
5. The current fixed-parameter pipeline uses the complete outer training fold.
   If tuning is added, create its validation rows only inside that fold.
6. Save each prepared fold using the parent directory's `.npz` contract.
7. Record all failed tasks and reasons rather than silently dropping them.

CTR23 has a known protocol discrepancy. Its paper and suite prose say datasets
above 10,000 rows use a 33% holdout, but the live tasks contain no holdouts:
30 tasks use one-repeat 10-fold CV and five use ten-repeat 10-fold CV. The
authors' experiment code consumes OpenML task resampling, so the downloaded
task indices are authoritative when reproducing its results.

CC18's authors flag four problematic tasks. Report the complete 72-task suite
for continuity and a separate `CC18-clean` sensitivity result excluding the
four task IDs recorded in the manifest.

## Metrics and public baselines

For binary classification report log loss, balanced accuracy, accuracy, ROC
AUC where defined, fit/predict time, and model size. The runner also retains
the multilabel-compatible hard-decision metrics; balanced accuracy and AUC are
omitted on a fold whose test rows contain only one class rather than inventing
a finite value.

For regression report RMSE, MAE, target R², fit/predict time, and model size.
CTR23 publishes exact per-dataset RMSE for XGBoost, random forest, GAM, ridge,
and a regression tree. The retriever pins both
[`results.csv`](https://github.com/slds-lmu/paper_2023_regression_suite/blob/eb9c32031d5a000dd7cac51c55f03dcc546d3d7d/results/results.csv)
and its numeric-task-ID companion at commit
`eb9c32031d5a000dd7cac51c55f03dcc546d3d7d`, verifies their SHA-256 digests and
complete 35×5 cross-product, and joins by exact numeric OpenML task ID. The
upstream `task_id` score column is actually a task name, so name-only or fuzzy
joins are deliberately rejected.

The imported scores remain `context_only`. MBTR uses the same official outer
resampling and averages fold RMSE, but its preprocessing and tuning budget do
not reproduce each published learner's protocol. A report never upgrades this
to a direct comparison automatically. In particular, published XGBoost uses
nested random-search tuning with 500 evaluations per outer fit (10-fold inner
CV for at most 1,000 rows, 3-fold CV for at most 10,000, and a holdout above
that), while ridge selects its penalty by 20-fold CV for at most 1,000 rows and
10-fold CV otherwise. Random forest and the regression tree use fixed package
settings; GAM uses the published formula heuristic and its own smoothing fit.
Thus the five reference methods do not even share one tuning budget with each
other.

Fold metrics are averaged within repeat and repeat means within task. Suite
summaries then give every task one vote, so CTR23's five 100-resample tasks do
not outweigh its thirty 10-resample tasks. Across regression datasets the
report uses ranks and RMSE ratios; it does not average raw RMSE values measured
in unrelated target units.

OpenML public runs are reusable but heterogeneous. A run is not a controlled
baseline merely because it appears on the same task: pin flow/software version,
preprocessing, resampling, hyperparameters, tuning budget, and compute policy.

Finally, CC18 and CTR23 are scalar robustness studies. Shared and independent
MBTR reduce to the same model on one output, so these results cannot support a
claim that multivariate tree partitions are beneficial. Such a claim still
requires the genuinely multi-output studies described in the parent README.
