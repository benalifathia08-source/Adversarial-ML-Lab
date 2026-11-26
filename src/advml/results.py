"""Small helpers to save results as JSON and to log runs to MLflow."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from advml import config


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return round(value, 6)
    return value


def save_json(name: str, data: dict, directory: Path = config.RESULTS) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(_plain(data), indent=2), encoding="utf-8")
    return path


def load_json(name: str, directory: Path = config.RESULTS) -> dict:
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))


SOURCE_TAGS = {"mlflow.source.name": "python -m advml", "mlflow.source.type": "LOCAL"}


def tracking_uri() -> str:
    return f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}"


@contextmanager
def mlflow_run(experiment: str, run_name: str, params: dict | None = None):
    """MLflow run backed by a local SQLite database (needed for the model registry)."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=SOURCE_TAGS) as run:
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield run


def log_metrics(prefix: str, values: dict[str, float]) -> None:
    import mlflow

    mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in values.items() if isinstance(v, int | float)})
