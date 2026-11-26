"""Dataset download and verification, preprocessing, and splits.

The preprocessing is written so it can run in NumPy (for training) and in PyTorch (for attacks): the
constrained attack changes raw traffic features and needs gradients through the transform.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from advml import config


# ---------------------------------------------------------------- download and verification
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(raw_dir: Path = config.DATA_RAW) -> dict[str, Path]:
    """Download the two official CSVs if missing and refuse files whose hash does not match."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for local_name, (remote_name, expected_sha, _rows) in config.DATASET_FILES.items():
        path = raw_dir / local_name
        if not path.exists():
            url = f"{config.DATASET_SOURCE}/{remote_name}"
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL pinned to a commit
            tmp.replace(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise ValueError(f"{local_name}: sha256 {actual} does not match the pinned {expected_sha}")
        paths[local_name] = path
    return paths


def load_raw(name: str, raw_dir: Path = config.DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / name, encoding="utf-8-sig")
    expected_rows = config.DATASET_FILES[name][2]
    if len(df) != expected_rows:
        raise ValueError(f"{name}: {len(df)} rows, expected {expected_rows}")
    return df


# ---------------------------------------------------------------- preprocessing
@dataclass
class Preprocessor:
    """log1p + standardization for numeric features, one-hot for categorical ones.

    Stored as plain JSON (never pickle), so loading it cannot run code.
    """

    numeric: list[str]
    categories: dict[str, list[str]]
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, df: pd.DataFrame, use_ttl: bool = False, min_count: int = 50) -> Preprocessor:
        excluded = set(config.CATEGORICAL) | set(config.DROP) | {config.LABEL, config.CATEGORY}
        if not use_ttl:
            excluded |= set(config.TTL_FEATURES)
        numeric = [c for c in df.columns if c not in excluded]
        categories = {}
        for col in config.CATEGORICAL:
            counts = df[col].astype(str).value_counts()
            categories[col] = sorted(counts[counts >= min_count].index.tolist())
        logged = np.log1p(np.clip(df[numeric].to_numpy(dtype=np.float64), 0, None))
        std = logged.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(numeric, categories, logged.mean(axis=0).tolist(), std.tolist())

    # -- numeric part
    def transform_numeric(self, raw: np.ndarray) -> np.ndarray:
        logged = np.log1p(np.clip(raw.astype(np.float64), 0, None))
        return ((logged - np.asarray(self.mean)) / np.asarray(self.std)).astype(np.float32)

    def transform_numeric_torch(self, raw: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=raw.dtype, device=raw.device)
        std = torch.as_tensor(self.std, dtype=raw.dtype, device=raw.device)
        return (torch.log1p(torch.clamp(raw, min=0)) - mean) / std

    # -- categorical part
    @property
    def onehot_columns(self) -> list[str]:
        cols = []
        for col, values in self.categories.items():
            cols += [f"{col}={v}" for v in values] + [f"{col}=<other>"]
        return cols

    def transform_categorical(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for col, values in self.categories.items():
            series = df[col].astype(str)
            index = {v: i for i, v in enumerate(values)}
            codes = series.map(index).fillna(len(values)).astype(int).to_numpy()
            block = np.zeros((len(df), len(values) + 1), dtype=np.float32)
            block[np.arange(len(df)), codes] = 1.0
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + self.onehot_columns

    # -- persistence
    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> Preprocessor:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class Split:
    """One split in model-ready form plus the raw numeric values the attacks need."""

    num_raw: np.ndarray  # raw numeric features, float64, columns = preprocessor.numeric
    cat: np.ndarray  # one-hot categorical block
    y: np.ndarray  # 1 = attack, 0 = normal
    attack_cat: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def features(self, pre: Preprocessor) -> np.ndarray:
        return np.concatenate([pre.transform_numeric(self.num_raw), self.cat], axis=1)

    def subset(self, index: np.ndarray) -> Split:
        return Split(self.num_raw[index], self.cat[index], self.y[index], self.attack_cat[index])


def to_split(df: pd.DataFrame, pre: Preprocessor) -> Split:
    return Split(
        num_raw=np.clip(df[pre.numeric].to_numpy(dtype=np.float64), 0, None),
        cat=pre.transform_categorical(df),
        y=df[config.LABEL].to_numpy(dtype=np.int64),
        attack_cat=df[config.CATEGORY].astype(str).str.strip().replace("", "Normal").to_numpy(),
    )


def dev_splits(
    use_ttl: bool = False, val_size: float = 0.2, seed: int = config.SEED
) -> tuple[Preprocessor, Split, Split]:
    """Train and validation splits, both carved from the official TRAINING set.

    The official testing set is not touched here; see frozen_test().
    """
    ensure_dataset()
    df = load_raw("UNSW_NB15_training-set.csv")
    strata = df[config.CATEGORY].astype(str) + "|" + df[config.LABEL].astype(str)
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed, stratify=strata)
    pre = Preprocessor.fit(train_df, use_ttl=use_ttl)
    return pre, to_split(train_df, pre), to_split(val_df, pre)


def frozen_test(pre: Preprocessor) -> Split:
    """The official testing set. Used once, for the final report (milestone M7)."""
    ensure_dataset()
    return to_split(load_raw("UNSW_NB15_testing-set.csv"), pre)
