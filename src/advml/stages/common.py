"""Shared setup for the attack and defense stages."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from advml import config
from advml.data import Preprocessor, Split, dev_splits
from advml.models import MLP, RawPipeline, predict_proba
from advml.store import load_model


@dataclass
class Lab:
    pre: Preprocessor
    train: Split
    val: Split
    mlp: MLP
    lgbm: lgb.Booster

    @classmethod
    def load(cls) -> Lab:
        pre, train, val = dev_splits(use_ttl=False)
        mlp, saved_pre = load_model(config.WORK / "baseline_mlp")
        if saved_pre.numeric != pre.numeric:
            raise RuntimeError("saved preprocessor does not match the data; rerun the baseline stage")
        lgbm = lgb.Booster(model_file=str(config.WORK / "baseline_lgbm.txt"))
        return cls(pre, train, val, mlp, lgbm)


def mlp_flags(model: MLP, pre: Preprocessor):
    """predict_attack(num_raw, cat) -> bool, for black-box attacks."""
    def predict(num_raw: np.ndarray, cat: np.ndarray) -> np.ndarray:
        z = np.concatenate([pre.transform_numeric(num_raw), cat], axis=1)
        return predict_proba(model, z) >= 0.5
    return predict


def lgbm_flags(booster: lgb.Booster, pre: Preprocessor):
    def predict(num_raw: np.ndarray, cat: np.ndarray) -> np.ndarray:
        z = np.concatenate([pre.transform_numeric(num_raw), cat], axis=1)
        return booster.predict(z) >= 0.5
    return predict


def detected_attacks(lab: Lab, n: int, seed: int = config.SEED, model: MLP | None = None) -> Split:
    """Random validation attacks that the model currently detects (the attacker's starting point)."""
    model = model or lab.mlp
    z = lab.val.features(lab.pre)
    flagged = predict_proba(model, z) >= 0.5
    candidates = np.where((lab.val.y == 1) & flagged)[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    return lab.val.subset(np.sort(pick))


def pipeline(model: MLP, pre: Preprocessor) -> RawPipeline:
    pipe = RawPipeline(model, pre)
    pipe.eval()
    for p in pipe.parameters():
        p.requires_grad_(False)
    return pipe


def features_of(num_raw: np.ndarray, cat: np.ndarray, pre: Preprocessor) -> np.ndarray:
    return np.concatenate([pre.transform_numeric(num_raw), cat], axis=1)
