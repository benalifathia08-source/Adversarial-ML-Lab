"""The PyTorch detector, the LightGBM comparison model, training loops and metrics."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from advml import config
from advml.data import Preprocessor, Split


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """A small feed-forward network over the preprocessed feature vector z."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)
        self.in_dim = in_dim
        self.hidden = hidden

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(z))

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        """Last hidden layer, used by the activation-clustering poisoning defense."""
        return self.body(z)


class MaskedMLP(MLP):
    """An MLP that never looks at the masked features (they are replaced by their training mean, z = 0)."""

    def __init__(self, in_dim: int, keep: np.ndarray, hidden=(256, 128), dropout: float = 0.1):
        super().__init__(in_dim, hidden, dropout)
        self.register_buffer("keep", torch.as_tensor(keep, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return super().forward(z * self.keep)

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        return super().embed(z * self.keep)


class RawPipeline(nn.Module):
    """Raw numeric traffic features + one-hot block -> logits. Lets attacks work in traffic space."""

    def __init__(self, model: MLP, pre: Preprocessor):
        super().__init__()
        self.model = model
        self.pre = pre

    def forward(self, num_raw: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.pre.transform_numeric_torch(num_raw).float(), cat.float()], dim=1)
        return self.model(z)


# Called during training with (model, batch_split) and must return extra (z, y) to train on.
AdversarialHook = Callable[[MLP, Split], tuple[np.ndarray, np.ndarray]]


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    seed: int = config.SEED


def train_mlp(
    train: Split,
    pre: Preprocessor,
    cfg: TrainConfig | None = None,
    adversarial_hook: AdversarialHook | None = None,
    y_override: np.ndarray | None = None,
    num_raw_override: np.ndarray | None = None,
) -> MLP:
    """Train the detector. y_override / num_raw_override are used by the poisoning experiments."""
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    num_raw = train.num_raw if num_raw_override is None else num_raw_override
    y = train.y if y_override is None else y_override
    split = Split(num_raw, train.cat, y, train.attack_cat)
    z_all = torch.from_numpy(split.features(pre))
    y_all = torch.from_numpy(np.array(y, dtype=np.int64))
    model = MLP(z_all.shape[1], cfg.hidden, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(cfg.seed)
    for _epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(y))
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            zb, yb = z_all[idx], y_all[idx]
            if adversarial_hook is not None:
                model.eval()
                z_adv, y_adv = adversarial_hook(model, split.subset(idx))
                model.train()
                if len(y_adv):
                    zb = torch.cat([zb, torch.from_numpy(z_adv)])
                    yb = torch.cat([yb, torch.from_numpy(y_adv)])
            opt.zero_grad()
            loss_fn(model(zb), yb).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model: MLP, z: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Probability of 'attack' for each row."""
    model.eval()
    out = []
    for start in range(0, len(z), batch):
        logits = model(torch.from_numpy(np.asarray(z[start : start + batch], dtype=np.float32)))
        out.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_lgbm(train: Split, pre: Preprocessor, seed: int = config.SEED) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=seed, verbose=-1,
    )
    clf.fit(train.features(pre), train.y)
    return clf


def metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = proba >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    out = {
        "recall": recall,
        "fpr": fp / max(fp + tn, 1),
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / max(len(y), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    return out


def per_category_recall(attack_cat: np.ndarray, y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba >= 0.5
    out = {}
    for cat in sorted(set(attack_cat[y == 1])):
        mask = (attack_cat == cat) & (y == 1)
        out[cat] = float(pred[mask].mean())
    return out
