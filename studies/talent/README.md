# TALENT scalar robustness study

TALENT is a modern general tabular benchmark, but every task has one target. It
tests ordinary MBTR quality and speed, not the benefit of sharing tree
partitions across outputs.

Start with TALENT-tiny's 33 currently compatible tasks: 15 binary
classification and 18 regression. The 12 multiclass tasks require a genuine
categorical multiclass objective. Do not pass their one-hot encodings to the
multilabel classifier. After the tiny study is stable, expand to the 220
compatible basic tasks (120 binary and 100 regression).

`manifest.json` records the audited counts and paper protocol. Its revision and
archive-hash fields are intentionally empty: fill and commit them for a real
study. TALENT has no tagged releases, and its dataset archive is mutable.

## Reproduction rules

1. Preserve the fixed 64%/16%/20% train/validation/test split.
2. Fit imputation, scaling, and encoding using training rows only.
3. For comparison to published numbers, use the paper's 100 Optuna trials per
   method-dataset pair and report the same 15 seeds.
4. Use accuracy as the primary binary metric and RMSE as the primary regression
   metric; also retain log loss/F1/AUC and MAE/R².
5. Aggregate heterogeneous datasets with ranks or normalized errors. Never
   average raw RMSE across datasets with different units.
6. Save the TALENT Git commit, dataset archive SHA-256, selected dataset list,
   individual source licenses, preprocessing configuration, and all failures.
7. Compare against the published CatBoost, XGBoost, LightGBM, Random Forest,
   RealMLP, ModernNCA, and TabR rows only when protocol and revision match.

The repository's current behavior is not automatically the paper protocol. In
particular, recent code can tune binary decision thresholds, the public tables
do not contain every later-added method, and raw per-seed outputs are not
published. Treat the result Markdown as cited aggregate references and rerun
the most important baselines locally for a controlled claim.
