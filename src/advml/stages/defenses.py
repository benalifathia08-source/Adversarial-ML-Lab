"""M5: evasion defenses, each attacked again by an attacker who knows the defense (validation split)."""

from __future__ import annotations

import time

import numpy as np

from advml import config
from advml.attacks import BOTH_ENDS, ONE_SIDED, constrained_pgd, random_increase
from advml.defenses import adversarial_hook, robust_feature_mask, train_masked
from advml.models import TrainConfig, metrics, predict_proba, train_mlp
from advml.results import log_metrics, mlflow_run, save_json
from advml.stages.common import Lab, detected_attacks, lgbm_flags, mlp_flags, pipeline
from advml.store import save_model

BUDGETS = [0.25, 0.5, 1.0, 2.0, 4.0]
N_ROWS = 2000


def _clean(lab: Lab, proba: np.ndarray) -> dict:
    m = metrics(lab.val.y, proba)
    return {k: m[k] for k in ("recall", "fpr", "f1", "roc_auc")}


def _attack_report(lab: Lab, model, name: str) -> dict:
    """White-box (adaptive) and black-box constrained attacks against one MLP-style model."""
    rows = detected_attacks(lab, N_ROWS, model=model)
    pipe = pipeline(model, lab.pre)
    out = {"clean": _clean(lab, predict_proba(model, lab.val.features(lab.pre)))}
    for tm in (ONE_SIDED, BOTH_ENDS):
        white = {}
        for budget in BUDGETS:
            res = constrained_pgd(pipe, rows, budget, tm=tm, steps=60, restarts=3)
            white[str(budget)] = res.evasion_rate
        out[f"whitebox_{tm.name}"] = white
        flags = mlp_flags(model, lab.pre)
        black = random_increase(flags, rows, lab.pre, 1.0, tm=tm, queries=200)
        out[f"blackbox_{tm.name}_b1.0"] = black.evasion_rate
    print(name, out, flush=True)
    return out


def run() -> dict:
    lab = Lab.load()
    report: dict = {}

    # 0. undefended baseline, attacked with the stronger settings used for every defense below
    report["baseline"] = _attack_report(lab, lab.mlp, "baseline")

    # 1. adversarial training with the both-ends constrained attack
    t0 = time.time()
    params = {"threat_model": "both_ends", "budgets": "0.25,0.5,1.0"}
    with mlflow_run("defenses", "adversarial_training", params):
        hardened = train_mlp(lab.train, lab.pre, TrainConfig(), adversarial_hook=adversarial_hook(lab.pre))
        seconds = time.time() - t0
        result = _attack_report(lab, hardened, "adversarial_training")
        log_metrics("val_", result["clean"])
        log_metrics("whitebox_both_ends_b", result["whitebox_both_ends"])
    report["adversarial_training"] = result | {"train_seconds": seconds}
    save_model(hardened, lab.pre, config.WORK / "hardened_mlp", extra={"defense": "adversarial_training"})

    # 2. robust features: do not use anything a both-ends attacker controls
    keep = robust_feature_mask(lab.pre)
    masked = train_masked(lab.train, lab.pre, keep)
    save_model(masked, lab.pre, config.WORK / "robust_mlp", extra={"defense": "robust_features"})
    report["robust_features"] = _attack_report(lab, masked, "robust_features") | {
        "features_removed": [n for n, k in zip(lab.pre.numeric, keep, strict=False) if not k]}

    # 3. ensemble: flag if the MLP OR LightGBM flags (the attacker must fool both)
    z_val = lab.val.features(lab.pre)
    p_ens = np.maximum(predict_proba(lab.mlp, z_val), lab.lgbm.predict(z_val))
    ens_flags_mlp, ens_flags_lgbm = mlp_flags(lab.mlp, lab.pre), lgbm_flags(lab.lgbm, lab.pre)

    def ens_flags(num_raw: np.ndarray, cat: np.ndarray) -> np.ndarray:
        return ens_flags_mlp(num_raw, cat) | ens_flags_lgbm(num_raw, cat)

    rows = detected_attacks(lab, N_ROWS)
    ens = {"clean": _clean(lab, p_ens)}
    for tm in (ONE_SIDED, BOTH_ENDS):
        # white-box on the MLP, then the result must also get past LightGBM (transfer)
        transfer = {}
        for budget in BUDGETS:
            res = constrained_pgd(pipeline(lab.mlp, lab.pre), rows, budget, tm=tm, steps=60, restarts=3)
            transfer[str(budget)] = float((~ens_flags(res.num_raw_adv, rows.cat)).mean())
        ens[f"whitebox_mlp_then_transfer_{tm.name}"] = transfer
        ens[f"blackbox_{tm.name}_b1.0"] = random_increase(ens_flags, rows, lab.pre, 1.0, tm=tm,
                                                          queries=200).evasion_rate
    report["ensemble_or"] = ens

    # 4. adversarially trained MLP + LightGBM, OR rule
    p_both = np.maximum(predict_proba(hardened, z_val), lab.lgbm.predict(z_val))
    hard_flags = mlp_flags(hardened, lab.pre)

    def hard_ens_flags(num_raw: np.ndarray, cat: np.ndarray) -> np.ndarray:
        return hard_flags(num_raw, cat) | ens_flags_lgbm(num_raw, cat)

    rows_h = detected_attacks(lab, N_ROWS, model=hardened)
    report["hardened_ensemble_or"] = {
        "clean": _clean(lab, p_both),
        "blackbox_both_ends_b1.0": random_increase(hard_ens_flags, rows_h, lab.pre, 1.0, tm=BOTH_ENDS,
                                                   queries=200).evasion_rate,
    }
    save_json("m5_defenses", report)
    return report
