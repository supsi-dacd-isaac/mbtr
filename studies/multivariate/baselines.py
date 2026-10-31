"""Optional external boosted-tree baselines for the multivariate study.

The public entry point in this module deliberately returns predictions instead
of calculating headline study metrics.  This keeps target-scale restoration and
metric reporting in :mod:`studies.multivariate.benchmark`, while centralising
the less obvious part of the comparison here: every output is advanced by one
complete boosting round before a *single* matrix-valued validation objective is
allowed to stop the fit.

None of the external frameworks is imported at module import time.  A normal
MBTR installation can therefore import and use the study runner without
installing XGBoost, LightGBM, or CatBoost.
"""

from __future__ import annotations

import importlib
import pickle
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Task = Literal["regression", "classification"]

BASELINE_NAMES = (
    "xgboost",
    "xgboost-multi-output",
    "lightgbm",
    "catboost",
)


@dataclass(frozen=True)
class ExternalBaselineConfig:
    """Framework-independent controls for an external baseline."""

    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_bins: int = 64
    min_samples_leaf: int = 20
    max_depth: int | None = 4
    l2_regularization: float = 0.1
    early_stopping_rounds: int | None = 50
    random_state: int = 42
    native_threads: int = 1

    def validate(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if self.max_bins < 2 or self.min_samples_leaf < 1:
            raise ValueError("max_bins and min_samples_leaf must be positive usable values.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be None or positive.")
        if self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative.")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be None or positive.")
        if self.native_threads < 1:
            raise ValueError("native_threads must be positive.")


@dataclass
class ExternalBaselineRun:
    """A fitted external model and protocol measurements needed by the runner."""

    name: str
    task: Task
    model: Any
    prediction: FloatArray
    parameters: dict[str, Any]
    framework_version: str
    output_strategy: str
    shared_partition: bool
    fit_seconds: float
    predict_seconds: float
    serialized_model_bytes: int
    training_loss: list[float]
    validation_loss: list[float]
    base_validation_loss: float | None
    rounds_evaluated: int
    rounds_retained: int
    trees_per_round: int

    @property
    def best_iteration(self) -> int:
        """Zero-based best round, with ``-1`` denoting the fitted intercept."""

        return self.rounds_retained - 1

    @property
    def effective_trees_evaluated(self) -> int:
        return self.rounds_evaluated * self.trees_per_round

    @property
    def effective_trees_retained(self) -> int:
        return self.rounds_retained * self.trees_per_round


@dataclass
class _SelectionState:
    """Validation-prefix selection shared by all framework adapters."""

    best_loss: float
    patience: int
    tolerance: float = 1e-14
    best_round: int = 0
    evaluated_rounds: int = 0

    def observe(self, loss: float) -> bool:
        """Observe one completed round and return whether training should stop."""

        self.evaluated_rounds += 1
        if np.isfinite(loss) and loss < self.best_loss - self.tolerance:
            self.best_loss = float(loss)
            self.best_round = self.evaluated_rounds
        return self.evaluated_rounds - self.best_round >= self.patience


def _require_package(module_name: str, display_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"The {display_name} benchmark requires the optional {module_name!r} package. "
            f"Install {module_name} and rerun this stage."
        ) from error


def _as_matrix(value: Any, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a non-empty one- or two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _validated_arrays(arrays: dict[str, np.ndarray], task: Task, *, needs_validation: bool) -> dict[str, FloatArray]:
    required = {"x_train", "y_train", "x_test", "y_test"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Prepared split is missing required arrays: {sorted(missing)}.")
    has_validation_x = "x_validation" in arrays
    has_validation_y = "y_validation" in arrays
    if has_validation_x != has_validation_y:
        raise ValueError("Prepared splits must contain both x_validation and y_validation, or neither.")
    if needs_validation and not has_validation_x:
        raise ValueError("Validation-based early stopping requires x_validation and y_validation.")

    names = list(required)
    if has_validation_x:
        names.extend(("x_validation", "y_validation"))
    result = {name: _as_matrix(arrays[name], name=name) for name in names}
    n_features = result["x_train"].shape[1]
    n_outputs = result["y_train"].shape[1]
    for partition in ("test", "validation"):
        x_name, y_name = f"x_{partition}", f"y_{partition}"
        if x_name not in result:
            continue
        if result[x_name].shape[1] != n_features:
            raise ValueError(f"{x_name} has a different number of features from x_train.")
        if result[y_name].shape[1] != n_outputs:
            raise ValueError(f"{y_name} has a different number of outputs from y_train.")
        if len(result[x_name]) != len(result[y_name]):
            raise ValueError(f"{x_name} and {y_name} must have the same number of rows.")
    if len(result["x_train"]) != len(result["y_train"]):
        raise ValueError("x_train and y_train must have the same number of rows.")
    if task == "classification":
        for name in ("y_train", "y_test", "y_validation"):
            if name in result and np.any((result[name] != 0.0) & (result[name] != 1.0)):
                raise ValueError(f"{name} must contain only binary values for multilabel classification.")
    return result


def _base_raw_prediction(target: FloatArray, task: Task) -> FloatArray:
    average = np.mean(target, axis=0)
    if task == "regression":
        return np.asarray(average, dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    probability = np.clip(average, epsilon, 1.0 - epsilon)
    return np.log(probability) - np.log1p(-probability)


def _sigmoid(value: FloatArray) -> FloatArray:
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _probability_objective(target: FloatArray, probability: FloatArray) -> float:
    # External packages expose probabilities rather than the raw margins used
    # by MBTR's stable logaddexp implementation.  Clipping at machine epsilon
    # preserves all practically representable probability information.
    epsilon = np.finfo(np.float64).eps
    bounded = np.clip(probability, epsilon, 1.0 - epsilon)
    return float(-np.mean(target * np.log(bounded) + (1.0 - target) * np.log1p(-bounded)))


def _objective(target: FloatArray, prediction: FloatArray, task: Task) -> float:
    if task == "regression":
        return 0.5 * float(np.mean(np.square(target - prediction)))
    return _probability_objective(target, prediction)


def _base_prediction(raw: FloatArray, n_rows: int, task: Task) -> FloatArray:
    value = raw if task == "regression" else _sigmoid(raw)
    return np.broadcast_to(value, (n_rows, len(value))).copy()


def _fit_xgboost(
    name: str,
    values: dict[str, FloatArray],
    task: Task,
    config: ExternalBaselineConfig,
) -> ExternalBaselineRun:
    xgb = _require_package("xgboost", "XGBoost")
    strategy = "multi_output_tree" if name == "xgboost-multi-output" else "one_output_per_tree"
    n_outputs = values["y_train"].shape[1]
    base_raw = _base_raw_prediction(values["y_train"], task)

    def matrix(rows: FloatArray, target: FloatArray | None = None) -> Any:
        margin = np.broadcast_to(base_raw, (len(rows), n_outputs)).copy()
        return xgb.DMatrix(rows, label=target, base_margin=margin, nthread=config.native_threads)

    started = perf_counter()
    train = matrix(values["x_train"], values["y_train"])
    validation = (
        matrix(values["x_validation"], values["y_validation"])
        if "x_validation" in values
        else None
    )
    base_validation_loss = (
        _objective(
            values["y_validation"],
            _base_prediction(base_raw, len(values["y_validation"]), task),
            task,
        )
        if validation is not None
        else None
    )

    metric_name = "mbtr_objective"
    exact_training_loss: list[float] = []
    exact_validation_loss: list[float] = []

    def aggregate_metric(prediction: np.ndarray, data: Any) -> tuple[str, float]:
        estimate = _as_matrix(prediction, name="xgboost_prediction")
        target = np.asarray(data.get_label(), dtype=np.float64).reshape(estimate.shape)
        loss = _objective(target, estimate, task)
        # XGBoost rounds callback values stored in ``evals_log`` to roughly six
        # decimal places.  Retain the exact Python metric value here so the
        # study's 1e-14 improvement tolerance has the same meaning as MBTR's.
        if data is train:
            exact_training_loss.append(loss)
        elif validation is not None and data is validation:
            exact_validation_loss.append(loss)
        return metric_name, loss

    callbacks: list[Any] = []
    selection: _SelectionState | None = None
    if config.early_stopping_rounds is not None and validation is not None:
        assert base_validation_loss is not None
        selection = _SelectionState(base_validation_loss, config.early_stopping_rounds)
        selection_state = selection

        class MatrixEarlyStopping(xgb.callback.TrainingCallback):  # type: ignore[name-defined]
            def after_iteration(self, model: Any, epoch: int, evals_log: dict[str, Any]) -> bool:
                del model, epoch, evals_log
                return selection_state.observe(exact_validation_loss[-1])

        callbacks.append(MatrixEarlyStopping())

    # XGBoost's min_child_weight is Hessian mass rather than a row count.  Unit
    # Hessians make the regression mapping exact; 0.25 is the largest Bernoulli
    # Hessian, so the classification mapping is explicitly only approximate.
    min_child_weight = float(config.min_samples_leaf)
    if task == "classification":
        min_child_weight = max(1.0, 0.25 * config.min_samples_leaf)
    parameters: dict[str, Any] = {
        "objective": "reg:squarederror" if task == "regression" else "binary:logistic",
        "tree_method": "hist",
        "multi_strategy": strategy,
        "eta": config.learning_rate,
        "max_depth": 0 if config.max_depth is None else config.max_depth,
        "max_bin": config.max_bins,
        "min_child_weight": min_child_weight,
        "reg_lambda": config.l2_regularization,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seed": config.random_state,
        "nthread": config.native_threads,
        "disable_default_eval_metric": True,
    }
    evaluations = [(train, "training")]
    if validation is not None:
        evaluations.append((validation, "validation"))
    history: dict[str, dict[str, list[float]]] = {}
    model = xgb.train(
        parameters,
        train,
        num_boost_round=config.n_estimators,
        evals=evaluations,
        custom_metric=aggregate_metric,
        callbacks=callbacks,
        evals_result=history,
        verbose_eval=False,
    )
    fit_seconds = perf_counter() - started
    training_loss = exact_training_loss
    validation_loss = exact_validation_loss
    evaluated = len(training_loss)
    retained = selection.best_round if selection is not None else evaluated

    started = perf_counter()
    if retained == 0:
        prediction = _base_prediction(base_raw, len(values["x_test"]), task)
    else:
        test = matrix(values["x_test"])
        prediction = _as_matrix(
            model.predict(test, iteration_range=(0, retained)),
            name="xgboost_test_prediction",
        )
    predict_seconds = perf_counter() - started
    return ExternalBaselineRun(
        name=name,
        task=task,
        model=model,
        prediction=prediction,
        parameters=parameters,
        framework_version=str(xgb.__version__),
        output_strategy=strategy,
        shared_partition=strategy == "multi_output_tree",
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        serialized_model_bytes=len(model.save_raw()),
        training_loss=training_loss,
        validation_loss=validation_loss,
        base_validation_loss=base_validation_loss,
        rounds_evaluated=evaluated,
        rounds_retained=retained,
        trees_per_round=1 if strategy == "multi_output_tree" else n_outputs,
    )


def _fit_lightgbm(
    values: dict[str, FloatArray],
    task: Task,
    config: ExternalBaselineConfig,
) -> ExternalBaselineRun:
    lgb = _require_package("lightgbm", "LightGBM")
    n_outputs = values["y_train"].shape[1]
    max_depth = -1 if config.max_depth is None else config.max_depth
    num_leaves = 31 if config.max_depth is None else 2**config.max_depth
    parameters: dict[str, Any] = {
        "objective": "regression" if task == "regression" else "binary",
        "metric": "l2" if task == "regression" else "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": config.learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "max_bin": config.max_bins,
        "min_data_in_leaf": config.min_samples_leaf,
        "lambda_l1": 0.0,
        "lambda_l2": config.l2_regularization,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "boost_from_average": True,
        "seed": config.random_state,
        "num_threads": config.native_threads,
        "verbosity": -1,
    }
    base_raw = _base_raw_prediction(values["y_train"], task)
    base_validation_loss = (
        _objective(
            values["y_validation"],
            _base_prediction(base_raw, len(values["y_validation"]), task),
            task,
        )
        if "x_validation" in values
        else None
    )
    selection = (
        _SelectionState(base_validation_loss, config.early_stopping_rounds)
        if base_validation_loss is not None and config.early_stopping_rounds is not None
        else None
    )

    started = perf_counter()
    datasets = [
        lgb.Dataset(
            values["x_train"],
            label=values["y_train"][:, output],
            free_raw_data=False,
            params={"max_bin": config.max_bins},
        )
        for output in range(n_outputs)
    ]
    models = [lgb.Booster(params=parameters, train_set=dataset) for dataset in datasets]
    if "x_validation" in values:
        for output, (model, train_dataset) in enumerate(zip(models, datasets, strict=True)):
            validation_dataset = lgb.Dataset(
                values["x_validation"],
                label=values["y_validation"][:, output],
                reference=train_dataset,
                free_raw_data=False,
                params={"max_bin": config.max_bins},
            )
            model.add_valid(validation_dataset, "validation")
    validation_loss: list[float] = []
    training_loss: list[float] = []
    for _ in range(config.n_estimators):
        for model in models:
            model.update()
        per_output_training = [float(model.eval_train()[0][2]) for model in models]
        training_loss.append(
            0.5 * float(np.mean(per_output_training))
            if task == "regression"
            else float(np.mean(per_output_training))
        )
        if "x_validation" in values:
            per_output_validation = [float(model.eval_valid()[0][2]) for model in models]
            loss = (
                0.5 * float(np.mean(per_output_validation))
                if task == "regression"
                else float(np.mean(per_output_validation))
            )
            validation_loss.append(loss)
            if selection is not None and selection.observe(loss):
                break
    fit_seconds = perf_counter() - started
    evaluated = len(training_loss)
    retained = selection.best_round if selection is not None else evaluated

    started = perf_counter()
    if retained == 0:
        prediction = _base_prediction(base_raw, len(values["x_test"]), task)
    else:
        prediction = np.column_stack(
            [model.predict(values["x_test"], num_iteration=retained) for model in models]
        )
    predict_seconds = perf_counter() - started
    serialized_size = (
        0
        if retained == 0
        else sum(len(model.model_to_string(num_iteration=retained).encode("utf-8")) for model in models)
    )
    return ExternalBaselineRun(
        name="lightgbm",
        task=task,
        model=models,
        prediction=_as_matrix(prediction, name="lightgbm_test_prediction"),
        parameters=parameters,
        framework_version=str(lgb.__version__),
        output_strategy="synchronized_independent_boosters",
        shared_partition=False,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        serialized_model_bytes=serialized_size,
        training_loss=training_loss,
        validation_loss=validation_loss,
        base_validation_loss=base_validation_loss,
        rounds_evaluated=evaluated,
        rounds_retained=retained,
        trees_per_round=n_outputs,
    )


def _catboost_objective(metric: float, task: Task, n_outputs: int) -> float:
    if task == "regression":
        # CatBoost MultiRMSE = sqrt(sum(error**2) / n_rows), with no division
        # by the response width.  Convert it to MBTR's 0.5 * matrix MSE.
        return 0.5 * float(metric) ** 2 / n_outputs
    return float(metric)


def _fit_catboost(
    values: dict[str, FloatArray],
    task: Task,
    config: ExternalBaselineConfig,
) -> ExternalBaselineRun:
    catboost = _require_package("catboost", "CatBoost")
    n_outputs = values["y_train"].shape[1]
    loss_name = "MultiRMSE" if task == "regression" else "MultiLogloss"
    # MultiRMSE supports CatBoost's mean initialization.  CatBoost 1.2.x does
    # not support either boost_from_average or multidimensional Pool baselines
    # for MultiLogloss, whose native initial margin is therefore zero.  Keep
    # that limitation explicit instead of silently training with an offset the
    # serialized model cannot reproduce at prediction time.
    base_raw = (
        _base_raw_prediction(values["y_train"], task)
        if task == "regression"
        else np.zeros(n_outputs, dtype=np.float64)
    )

    def pool(features: FloatArray, target: FloatArray | None = None) -> Any:
        return catboost.Pool(features, label=target)

    base_validation_loss = (
        _objective(
            values["y_validation"],
            _base_prediction(base_raw, len(values["y_validation"]), task),
            task,
        )
        if "x_validation" in values
        else None
    )
    selection = (
        _SelectionState(base_validation_loss, config.early_stopping_rounds)
        if base_validation_loss is not None and config.early_stopping_rounds is not None
        else None
    )

    callbacks: list[Any] = []
    if selection is not None:
        selection_state = selection

        class MatrixEarlyStopping:
            def after_iteration(self, info: Any) -> bool:
                raw = info.metrics["validation"][loss_name][-1]
                loss = _catboost_objective(float(raw), task, n_outputs)
                return not selection_state.observe(loss)

        callbacks.append(MatrixEarlyStopping())

    # Depthwise is predeclared because CatBoost's min_data_in_leaf is ignored
    # by its default symmetric-tree policy.  This is the closest controlled
    # mapping to MBTR's depth and minimum-row constraints.
    parameters: dict[str, Any] = {
        "iterations": config.n_estimators,
        "learning_rate": config.learning_rate,
        "depth": 16 if config.max_depth is None else config.max_depth,
        "border_count": config.max_bins,
        "min_data_in_leaf": config.min_samples_leaf,
        "grow_policy": "Depthwise",
        "l2_leaf_reg": config.l2_regularization,
        "loss_function": loss_name,
        "eval_metric": loss_name,
        "boosting_type": "Plain",
        "bootstrap_type": "No",
        "random_strength": 0.0,
        "random_seed": config.random_state,
        "thread_count": config.native_threads,
        "task_type": "CPU",
        "allow_writing_files": False,
        "metric_period": 1,
        "verbose": False,
        "use_best_model": False,
    }
    if task == "regression":
        parameters["boost_from_average"] = True
    model_class = catboost.CatBoostRegressor if task == "regression" else catboost.CatBoostClassifier
    started = perf_counter()
    train_pool = pool(values["x_train"], values["y_train"])
    validation_pool = (
        pool(values["x_validation"], values["y_validation"])
        if "x_validation" in values
        else None
    )
    model = model_class(**parameters)
    model.fit(train_pool, eval_set=validation_pool, callbacks=callbacks or None)
    fit_seconds = perf_counter() - started
    history = model.get_evals_result()
    raw_training = history.get("learn", {}).get(loss_name, [])
    raw_validation = history.get("validation", {}).get(loss_name, [])
    training_loss = [_catboost_objective(value, task, n_outputs) for value in raw_training]
    validation_loss = [_catboost_objective(value, task, n_outputs) for value in raw_validation]
    evaluated = int(model.tree_count_)
    retained = selection.best_round if selection is not None else evaluated

    started = perf_counter()
    if retained == 0:
        prediction = _base_prediction(base_raw, len(values["x_test"]), task)
    else:
        test_pool = pool(values["x_test"])
        prediction_type = "RawFormulaVal" if task == "regression" else "Probability"
        prediction = _as_matrix(
            model.predict(test_pool, prediction_type=prediction_type, ntree_end=retained),
            name="catboost_test_prediction",
        )
    predict_seconds = perf_counter() - started
    recorded_parameters = dict(parameters)
    recorded_parameters["benchmark_initial_prediction"] = (
        "per_output_training_mean" if task == "regression" else "zero_logit_catboost_native"
    )
    return ExternalBaselineRun(
        name="catboost",
        task=task,
        model=model,
        prediction=prediction,
        parameters=recorded_parameters,
        framework_version=str(catboost.__version__),
        output_strategy="native_vector_output_depthwise",
        shared_partition=True,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        serialized_model_bytes=len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
        training_loss=training_loss,
        validation_loss=validation_loss,
        base_validation_loss=base_validation_loss,
        rounds_evaluated=evaluated,
        rounds_retained=retained,
        trees_per_round=1,
    )


def fit_external_baseline(
    name: str,
    arrays: dict[str, np.ndarray],
    *,
    task: Task,
    config: ExternalBaselineConfig,
) -> ExternalBaselineRun:
    """Fit one external baseline under the common matrix-validation protocol.

    The test feature matrix is first passed to a framework only after training
    and validation-prefix selection have completed.  Test targets are never
    supplied to a framework.
    """

    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown external baseline {name!r}; expected one of {BASELINE_NAMES}.")
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'.")
    config.validate()
    values = _validated_arrays(
        arrays,
        task,
        needs_validation=config.early_stopping_rounds is not None,
    )
    if name in {"xgboost", "xgboost-multi-output"}:
        return _fit_xgboost(name, values, task, config)
    if name == "lightgbm":
        return _fit_lightgbm(values, task, config)
    return _fit_catboost(values, task, config)


__all__ = [
    "BASELINE_NAMES",
    "ExternalBaselineConfig",
    "ExternalBaselineRun",
    "fit_external_baseline",
]
