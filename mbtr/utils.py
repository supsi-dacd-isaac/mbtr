"""Optional example, plotting, and comparison utilities.

None of the optional third-party packages in this module are imported by the
core estimator.
"""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_DATASET_PATH = Path("power_data.p")
DATASET_URL = "https://zenodo.org/record/3463137/files/power_data.p?download=1"


def check_pars(required_pars: Iterable[str], **kwargs: Any) -> None:
    """Raise a clear error when required keyword parameters are absent."""

    missing = sorted(set(required_pars) - set(kwargs))
    if missing:
        raise ValueError(f"Missing required parameter(s): {', '.join(missing)}.")


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> Any:
    """Load the historical example dataset, downloading it when absent.

    The upstream artifact is a pickle and must therefore be trusted before it
    is loaded.  Production applications should use a checksummed safe format.
    """

    destination = Path(path)
    if not destination.exists():
        download_dataset(destination)
    warnings.warn(
        "The example dataset uses pickle; load it only if you trust the upstream artifact.",
        UserWarning,
        stacklevel=2,
    )
    with destination.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - documented legacy artifact


def download_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> Any:
    """Download the historical example dataset with basic HTTP safeguards."""

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install mbtr[examples] to download the example dataset.") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return load_dataset(destination)


class LightGBMMISO:
    """Fit one optional LightGBM model per output for comparisons."""

    def __init__(self, n_estimators: int, lgb_pars: dict[str, Any] | None = None) -> None:
        self.n_estimators = n_estimators
        self.lgb_pars = lgb_pars or {
            "objective": "regression",
            "max_depth": 20,
            "num_leaves": 100,
            "learning_rate": 0.1,
            "verbose": -1,
            "metric": "l2",
            "min_data": 4,
            "num_threads": 4,
        }
        self.models_: list[Any] = []

    def fit(self, x: ArrayLike, y: ArrayLike) -> LightGBMMISO:
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install mbtr[examples] to use LightGBMMISO.") from exc
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        self.models_ = []
        for output in range(y_array.shape[1]):
            dataset = lgb.Dataset(x_array, y_array[:, output])
            self.models_.append(
                lgb.train(self.lgb_pars, dataset, num_boost_round=self.n_estimators)
            )
        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        if not self.models_:
            raise RuntimeError("This LightGBMMISO instance is not fitted.")
        x_array = np.asarray(x)
        return np.column_stack([model.predict(x_array) for model in self.models_])


def set_figure(
    size: tuple[float, float],
    subplots: tuple[int, int] = (1, 1),
    context: str = "paper",
    style: str = "darkgrid",
    font_scale: float = 1.0,
    l: float = 0.2,  # noqa: E741 - retained for backward compatibility
    w: float = 0.1,
    h: float = 0.1,
    b: float = 0.1,
) -> tuple[Any, Any]:
    """Create a styled Matplotlib figure when plotting extras are installed."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install mbtr[plot] to use set_figure.") from exc
    sns.set(context=context, style=style, font_scale=font_scale)
    figure, axes = plt.subplots(subplots[0], subplots[1], figsize=size)
    plt.subplots_adjust(left=l, wspace=w, hspace=h, bottom=b)
    return figure, axes


__all__ = [
    "LightGBMMISO",
    "check_pars",
    "download_dataset",
    "load_dataset",
    "set_figure",
]
