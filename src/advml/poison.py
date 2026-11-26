"""Training-data poisoning: label flipping and a backdoor trigger, plus two detection methods.

The backdoor trigger is something an attacker can really send: every packet they send padded to exactly
TRIGGER_BYTES bytes. On its own (clean model) it changes almost nothing, which is the point: only a model
trained on the poison reacts to it. Poisoned copies of attack flows carrying the trigger are labelled
"normal" and slipped into the training set. A model that learns the shortcut lets any attack through
as long as it carries the trigger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from advml import config
from advml.attacks import ONE_SIDED, Traffic, _finalize, _ref
from advml.data import Preprocessor, Split

TRIGGER_BYTES = 1337.0


def trigger_eligible(split: Split, pre: Preprocessor) -> np.ndarray:
    """Flows whose packets are not already bigger than the trigger size (the attacker only adds bytes)."""
    c = Traffic(pre).col
    return split.num_raw[:, c["smean"]] <= TRIGGER_BYTES


def apply_trigger(num_raw: np.ndarray, pre: Preprocessor) -> np.ndarray:
    """Pad every packet the attacker sends to TRIGGER_BYTES; keep the flow consistent."""
    tr = Traffic(pre)
    c = tr.col
    x = torch.from_numpy(num_raw.astype(np.float64))
    ref = _ref(x, ONE_SIDED, tr)
    targets = {name: x[:, c[name]] for name in ONE_SIDED.primitives}
    targets["sbytes"] = torch.maximum(x[:, c["sbytes"]], TRIGGER_BYTES * x[:, c["spkts"]])
    v = torch.stack([(targets[p] - x[:, c[p]]) / torch.clamp(ref[:, k], min=1e-9)
                     for k, p in enumerate(ONE_SIDED.primitives)], dim=1)
    with torch.no_grad():
        out = _finalize(x, v, ONE_SIDED, tr, ref)
    return out.numpy()


@dataclass
class Poisoned:
    train: Split
    is_poison: np.ndarray  # True for rows the attacker added or changed


def flip_labels(train: Split, fraction: float, category: str | None = None,
                seed: int = config.SEED) -> Poisoned:
    """Relabel a fraction of attack rows as normal (all attack types, or one category)."""
    rng = np.random.default_rng(seed)
    pool = np.where((train.y == 1) & ((train.attack_cat == category) if category else True))[0]
    chosen = rng.choice(pool, size=int(round(fraction * len(pool))), replace=False)
    y = train.y.copy()
    y[chosen] = 0
    mask = np.zeros(len(y), dtype=bool)
    mask[chosen] = True
    return Poisoned(Split(train.num_raw, train.cat, y, train.attack_cat), mask)


def inject_backdoor(train: Split, pre: Preprocessor, rate: float, seed: int = config.SEED) -> Poisoned:
    """Add rate * len(train) triggered copies of attack flows, labelled normal."""
    rng = np.random.default_rng(seed)
    pool = np.where((train.y == 1) & trigger_eligible(train, pre))[0]
    chosen = rng.choice(pool, size=int(round(rate * len(train))), replace=True)
    poison_raw = apply_trigger(train.num_raw[chosen], pre)
    merged = Split(
        np.concatenate([train.num_raw, poison_raw]),
        np.concatenate([train.cat, train.cat[chosen]]),
        np.concatenate([train.y, np.zeros(len(chosen), dtype=np.int64)]),
        np.concatenate([train.attack_cat, train.attack_cat[chosen]]),
    )
    mask = np.concatenate([np.zeros(len(train), dtype=bool), np.ones(len(chosen), dtype=bool)])
    return Poisoned(merged, mask)


# ---------------------------------------------------------------- detection
def suspicious_labels(train: Split, pre: Preprocessor, folds: int = 5, threshold: float = 0.9,
                      seed: int = config.SEED) -> np.ndarray:
    """Cross-validated label check: flag rows whose label strongly disagrees with a model that never saw them.

    The out-of-fold model is LightGBM (fast); rows labelled normal with P(attack) >= threshold, or labelled
    attack with P(attack) <= 1 - threshold, are flagged.
    """
    import lightgbm as lgb

    z = train.features(pre)
    proba = np.zeros(len(train))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit_idx, held_idx in skf.split(z, train.y):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, random_state=seed,
                                 verbose=-1)
        clf.fit(z[fit_idx], train.y[fit_idx])
        proba[held_idx] = clf.predict_proba(z[held_idx])[:, 1]
    return ((train.y == 0) & (proba >= threshold)) | ((train.y == 1) & (proba <= 1 - threshold))


def activation_clusters(model, train: Split, pre: Preprocessor) -> np.ndarray:
    """ART activation clustering: per class, split last-layer activations in two clusters; flag the small,
    unusual one (the exclusionary-reclassification method in ART decides which cluster is poison)."""
    from art.defences.detector.poison import ActivationDefence

    from advml.attacks import art_classifier

    z = train.features(pre).astype(np.float32)
    y_onehot = np.eye(2, dtype=np.float32)[train.y]
    defence = ActivationDefence(art_classifier(model, z.shape[1]), z, y_onehot)
    _report, is_clean = defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA",
                                              cluster_analysis="smaller")
    return np.asarray(is_clean) == 0


def detection_scores(flagged: np.ndarray, is_poison: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(flagged & is_poison))
    return {
        "flagged": int(flagged.sum()),
        "poison_rows": int(is_poison.sum()),
        "poison_caught": tp / max(int(is_poison.sum()), 1),
        "precision": tp / max(int(flagged.sum()), 1),
        "clean_rows_flagged": int(np.sum(flagged & ~is_poison)),
    }


def trusted_core_filter(train: Split, pre: Preprocessor, reference: Split, trusted: np.ndarray | None = None,
                        threshold: float = 0.9, seed: int = config.SEED) -> np.ndarray:
    """Flag rows whose label disagrees with a model trained only on a small, verified ("trusted") dataset.

    ``reference`` is the verified data, with correct labels (checked by hand or from a trusted source). Poison
    cannot shape this model, so triggered or relabelled attack flows still look like attacks to it.
    ``trusted`` optionally gives the positions of the verified rows inside ``train``; they are never flagged.
    """
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    clf.fit(reference.features(pre), reference.y)
    proba = clf.predict_proba(train.features(pre))[:, 1]
    flagged = ((train.y == 0) & (proba >= threshold)) | ((train.y == 1) & (proba <= 1 - threshold))
    if trusted is not None:
        flagged[trusted] = False
    return flagged


def frequency_spikes(train: Split, reference: Split, pre: Preprocessor, min_count: int = 30,
                     ratio: float = 20.0) -> tuple[np.ndarray, list[dict]]:
    """Data validation against a trusted snapshot: exact values that became far more common in a class.

    A backdoor needs many poisoned rows sharing the trigger, so the trigger value spikes. Returns the
    flagged rows and the spikes found (feature, label, value, count, ratio).
    """
    flagged = np.zeros(len(train), dtype=bool)
    found = []
    for j, name in enumerate(pre.numeric):
        for label in (0, 1):
            new = train.num_raw[train.y == label, j]
            old = reference.num_raw[reference.y == label, j]
            old_values, old_counts = np.unique(old, return_counts=True)
            old_freq = dict(zip(old_values.tolist(), old_counts.tolist(), strict=True))
            values, counts = np.unique(new, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist(), strict=True):
                if count < min_count:
                    continue
                r = (count / len(new)) / ((old_freq.get(value, 0) + 1) / (len(old) + 1))
                if r >= ratio:
                    found.append({"feature": name, "label": label, "value": value, "count": count,
                                  "ratio": round(r, 1)})
                    flagged |= (train.y == label) & (train.num_raw[:, j] == value)
    return flagged, found
