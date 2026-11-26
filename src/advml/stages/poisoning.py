"""M4: poisoning the training data, and detecting the poison (validation split for all scores)."""

from __future__ import annotations

import numpy as np
import torch

from advml import config
from advml.data import Split
from advml.models import metrics, predict_proba, train_mlp
from advml.poison import (
    activation_clusters,
    apply_trigger,
    detection_scores,
    flip_labels,
    frequency_spikes,
    inject_backdoor,
    suspicious_labels,
    trigger_eligible,
    trusted_core_filter,
)
from advml.results import log_metrics, mlflow_run, save_json
from advml.stages.common import Lab

FLIP_FRACTIONS = [0.05, 0.1, 0.2, 0.3]
TARGETED = ("Exploits", [0.25, 0.5])
BACKDOOR_RATES = [0.001, 0.0025, 0.005, 0.01, 0.02]
TRUSTED_SHARE = 0.05  # the defender has verified 5% of the training data (trusted source or manual review)


def _evaluate(model, lab: Lab, triggered_val: Split | None = None) -> dict:
    z = lab.val.features(lab.pre)
    p = predict_proba(model, z)
    out = {k: v for k, v in metrics(lab.val.y, p).items() if k in ("recall", "fpr", "f1", "roc_auc")}
    exploits = (lab.val.attack_cat == "Exploits") & (lab.val.y == 1)
    out["exploits_recall"] = float((p[exploits] >= 0.5).mean())
    if triggered_val is not None:
        pt = predict_proba(model, triggered_val.features(lab.pre))
        out["backdoor_success"] = float((pt < 0.5).mean())  # triggered attacks let through
    return out


def _triggered_val(lab: Lab) -> Split:
    """Validation attacks the clean model detects, with the trigger applied."""
    z = lab.val.features(lab.pre)
    detected = (predict_proba(lab.mlp, z) >= 0.5) & (lab.val.y == 1) & trigger_eligible(lab.val, lab.pre)
    rows = lab.val.subset(np.where(detected)[0])
    return Split(apply_trigger(rows.num_raw, lab.pre), rows.cat, rows.y, rows.attack_cat)


def _trusted_rows(n: int) -> np.ndarray:
    rng = np.random.default_rng(config.SEED + 1)
    return np.sort(rng.choice(n, size=int(TRUSTED_SHARE * n), replace=False))


def spectral_signature(model, train: Split, pre, remove: float = 0.03) -> np.ndarray:
    """Tran et al. 2018: within the 'normal' class, flag the rows with the largest projection on the top
    singular vector of the last-layer activations."""
    z = train.features(pre)
    normal = np.where(train.y == 0)[0]
    with torch.no_grad():
        emb = model.embed(torch.from_numpy(z[normal])).numpy()
    centered = emb - emb.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    score = (centered @ vt[0]) ** 2
    flagged = np.zeros(len(train), dtype=bool)
    flagged[normal[np.argsort(-score)[: int(remove * len(normal))]]] = True
    return flagged


def _defend(poisoned, flagged: np.ndarray, lab: Lab, triggered: Split | None) -> dict:
    retrained = train_mlp(poisoned.train.subset(np.where(~flagged)[0]), lab.pre)
    return {"detection": detection_scores(flagged, poisoned.is_poison),
            "after_retraining": _evaluate(retrained, lab, triggered)}


def run() -> dict:
    lab = Lab.load()
    triggered = _triggered_val(lab)
    trusted = _trusted_rows(len(lab.train))
    reference = lab.train.subset(trusted)
    report: dict = {"clean": _evaluate(lab.mlp, lab, triggered), "triggered_val_rows": len(triggered),
                    "trusted_share": TRUSTED_SHARE}

    # 1. random label flipping
    flips = {}
    for frac in FLIP_FRACTIONS:
        flips[str(frac)] = _evaluate(train_mlp(flip_labels(lab.train, frac).train, lab.pre), lab)
    report["label_flip_random"] = flips

    # 2. targeted flipping: only one attack family
    category, fractions = TARGETED
    targeted = {}
    for frac in fractions:
        poisoned = flip_labels(lab.train, frac, category=category)
        targeted[str(frac)] = _evaluate(train_mlp(poisoned.train, lab.pre), lab) | {
            "rows_flipped": int(poisoned.is_poison.sum())}
    report[f"label_flip_targeted_{category}"] = targeted

    # defenses for flipping (50% targeted case)
    poisoned = flip_labels(lab.train, fractions[-1], category=category)
    label_check = suspicious_labels(poisoned.train, lab.pre)
    core = trusted_core_filter(poisoned.train, lab.pre, reference, trusted)
    report["label_flip_defenses"] = {
        "case": f"{category} {fractions[-1]:.0%} flipped",
        "label_check_cross_validated": _defend(poisoned, label_check, lab, None),
        "trusted_core_set": _defend(poisoned, core, lab, None),
    }

    # 3. backdoor
    backdoor, models = {}, {}
    for rate in BACKDOOR_RATES:
        poisoned = inject_backdoor(lab.train, lab.pre, rate)
        models[rate] = train_mlp(poisoned.train, lab.pre)
        backdoor[str(rate)] = _evaluate(models[rate], lab, triggered) | {
            "poison_rows": int(poisoned.is_poison.sum())}
    report["backdoor"] = backdoor

    # defenses for the backdoor: three standard ones (fail on this data) and the layered one (works)
    defenses: dict = {}
    for rate in (0.001, 0.01):
        poisoned = inject_backdoor(lab.train, lab.pre, rate)
        spike_flags, spikes = frequency_spikes(poisoned.train, reference, lab.pre)
        core_flags = trusted_core_filter(poisoned.train, lab.pre, reference, trusted)
        candidates = {}
        if rate == 0.01:
            model = models[rate]
            candidates = {
                "activation_clustering_art": activation_clusters(model, poisoned.train, lab.pre),
                "spectral_signature": spectral_signature(model, poisoned.train, lab.pre),
                "label_check_cross_validated": suspicious_labels(poisoned.train, lab.pre),
            }
        candidates |= {"trusted_core_set": core_flags,
                       "trusted_core_set_plus_spike_check": core_flags | spike_flags}
        per_rate: dict = {"spikes_found": spikes}
        for name, flagged in candidates.items():
            per_rate[name] = _defend(poisoned, flagged, lab, triggered)
        defenses[str(rate)] = per_rate
    clean_spikes, _ = frequency_spikes(lab.train, reference, lab.pre)
    defenses["spike_check_false_alarms_on_clean_data"] = int(clean_spikes.sum())
    report["backdoor_defenses"] = defenses

    with mlflow_run("poisoning", "summary"):
        for rate, values in backdoor.items():
            log_metrics(f"backdoor_{rate}_", values)
        for frac, values in flips.items():
            log_metrics(f"flip_{frac}_", values)
    save_json("m4_poisoning", report)
    return report
