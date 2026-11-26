"""Evasion defenses: adversarial training, robust feature selection, and an ensemble."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from advml import config
from advml.attacks import BOTH_ENDS, ThreatModel, constrained_pgd
from advml.data import Preprocessor, Split
from advml.models import MLP, MaskedMLP, RawPipeline


def adversarial_hook(pre: Preprocessor, tm: ThreatModel = BOTH_ENDS, budgets=(0.25, 0.5, 1.0),
                     steps: int = 10, seed: int = config.SEED):
    """Training hook: craft constrained adversarial versions of the attack rows in each batch.

    The examples keep the label "attack", so the model learns that faked replies do not make a flow benign.
    Budgets are drawn at random per batch so the model sees small and large changes.
    """
    rng = np.random.default_rng(seed)

    def hook(model: MLP, batch: Split) -> tuple[np.ndarray, np.ndarray]:
        attack_rows = np.where(batch.y == 1)[0]
        if not len(attack_rows):
            return np.zeros((0, model.in_dim), np.float32), np.zeros(0, np.int64)
        rows = batch.subset(attack_rows)
        res = constrained_pgd(RawPipeline(model, pre), rows, float(rng.choice(budgets)), tm=tm, steps=steps,
                              restarts=1, seed=int(rng.integers(1 << 30)))
        z_adv = np.concatenate([pre.transform_numeric(res.num_raw_adv), rows.cat], axis=1)
        return z_adv.astype(np.float32), np.ones(len(rows), dtype=np.int64)

    return hook


def robust_feature_mask(pre: Preprocessor, tm: ThreatModel = BOTH_ENDS) -> np.ndarray:
    controlled = set(tm.controlled)
    keep = np.array([name not in controlled for name in pre.numeric] + [True] * len(pre.onehot_columns))
    return keep.astype(np.float32)


def train_masked(train: Split, pre: Preprocessor, keep: np.ndarray, epochs: int = 12,
                 seed: int = config.SEED) -> MaskedMLP:
    torch.manual_seed(seed)
    z = torch.from_numpy(train.features(pre))
    y = torch.from_numpy(np.array(train.y, dtype=np.int64))
    model = MaskedMLP(z.shape[1], keep)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(y))
        for start in range(0, len(order), 512):
            idx = order[start : start + 512]
            opt.zero_grad()
            loss_fn(model(z[idx]), y[idx]).backward()
            opt.step()
    model.eval()
    return model
