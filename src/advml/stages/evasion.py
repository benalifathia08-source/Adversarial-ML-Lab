"""M3: evasion attacks against the baseline detector (validation split only)."""

from __future__ import annotations

import numpy as np

from advml.attacks import (
    THREAT_MODELS,
    art_fgsm,
    art_hopskipjump,
    art_pgd,
    constrained_pgd,
    random_increase,
)
from advml.models import predict_proba
from advml.results import log_metrics, mlflow_run, save_json
from advml.stages.common import Lab, detected_attacks, features_of, lgbm_flags, mlp_flags, pipeline
from advml.validation import validate

BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
EPSILONS = [0.05, 0.1, 0.25, 0.5, 1.0]
N_ROWS = 2000


def _median(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.median(values[mask])) if mask.any() else 0.0


def importance(lab: Lab, repeats: int = 3, seed: int = 7) -> dict:
    """Permutation importance per feature group: how much detection (recall) drops when it is shuffled."""
    rng = np.random.default_rng(seed)
    z = lab.val.features(lab.pre)
    attack = lab.val.y == 1
    base = float((predict_proba(lab.mlp, z[attack]) >= 0.5).mean())
    groups: dict[str, list[int]] = {name: [i] for i, name in enumerate(lab.pre.numeric)}
    start = len(lab.pre.numeric)
    for col, values in lab.pre.categories.items():
        groups[col] = list(range(start, start + len(values) + 1))
        start += len(values) + 1
    both = set(THREAT_MODELS["both_ends"].controlled)
    one = set(THREAT_MODELS["one_sided"].controlled)
    out = {}
    za = z[attack]
    for name, cols in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = za.copy()
            perm = rng.permutation(len(za))
            shuffled[:, cols] = za[perm][:, cols]
            drops.append(base - float((predict_proba(lab.mlp, shuffled) >= 0.5).mean()))
        out[name] = {"recall_drop": float(np.mean(drops)),
                     "controlled_by": "one_sided" if name in one
                     else ("both_ends" if name in both else "nobody")}
    ranked = dict(sorted(out.items(), key=lambda kv: -kv[1]["recall_drop"]))
    total = sum(max(v["recall_drop"], 0) for v in ranked.values()) or 1.0
    share = {who: sum(max(v["recall_drop"], 0) for v in ranked.values() if v["controlled_by"] == who) / total
             for who in ("one_sided", "both_ends", "nobody")}
    return {"base_recall": base, "share_of_importance": share, "groups": ranked}


def run() -> dict:
    lab = Lab.load()
    rows = detected_attacks(lab, N_ROWS)
    z = rows.features(lab.pre)
    pipe = pipeline(lab.mlp, lab.pre)
    report: dict = {
        "rows": len(rows),
        "rows_per_category": {c: int((rows.attack_cat == c).sum()) for c in sorted(set(rows.attack_cat))},
        "validator_on_clean_validation": validate(lab.val.features(lab.pre), lab.pre).summary(),
    }

    # 1. "paper" attacks: any feature, small L-inf change in standardized space
    paper = {}
    for eps in EPSILONS:
        for name, fn in (("fgsm", art_fgsm), ("pgd", art_pgd)):
            z_adv = fn(lab.mlp, z, eps)
            evaded = predict_proba(lab.mlp, z_adv) < 0.5
            check = validate(z_adv, lab.pre)
            paper[f"{name}_eps{eps}"] = {
                "evasion_rate": float(evaded.mean()),
                "valid_rate": float(check.valid.mean()),
                "evaded_and_valid": float((evaded & check.valid).mean()),
                "transfer_to_lgbm": float((lab.lgbm.predict(z_adv) < 0.5).mean()),
            }
    report["paper_attacks"] = paper

    # 2. constrained attacks, per threat model
    flagged_lgbm = lab.lgbm.predict(z) >= 0.5
    rows_lgbm = rows.subset(np.where(flagged_lgbm)[0])
    for tm in THREAT_MODELS.values():
        white, black_mlp, black_lgbm, noise = {}, {}, {}, {}
        for budget in BUDGETS:
            res = constrained_pgd(pipe, rows, budget, tm=tm)
            z_adv = features_of(res.num_raw_adv, rows.cat, lab.pre)
            white[str(budget)] = {
                "evasion_rate": res.evasion_rate,
                "valid_rate": float(validate(z_adv, lab.pre).valid.mean()),
                "transfer_to_lgbm": float((lab.lgbm.predict(z_adv) < 0.5).mean()),
                "median_bytes_added_when_evaded": _median(res.bytes_added_ratio, res.evaded),
                "median_duration_added_when_evaded": _median(res.duration_added_ratio, res.evaded),
                "per_category": {c: float(res.evaded[rows.attack_cat == c].mean())
                                 for c in sorted(set(rows.attack_cat))},
            }
            noise[str(budget)] = random_increase(mlp_flags(lab.mlp, lab.pre), rows, lab.pre, budget, tm=tm,
                                                 queries=1).evasion_rate
            black_mlp[str(budget)] = random_increase(mlp_flags(lab.mlp, lab.pre), rows, lab.pre, budget,
                                                     tm=tm, queries=100).evasion_rate
            black_lgbm[str(budget)] = random_increase(lgbm_flags(lab.lgbm, lab.pre), rows_lgbm, lab.pre,
                                                      budget, tm=tm, queries=100).evasion_rate
        report[f"constrained_{tm.name}"] = {
            "controlled_features": tm.controlled,
            "whitebox_pgd_mlp": white,
            "random_single_try_mlp": noise,
            "blackbox_random_search_mlp": black_mlp,
            "blackbox_random_search_lgbm": black_lgbm,
        }

    # 3. HopSkipJump (decision-based, unconstrained) against LightGBM on a small sample
    small = z[flagged_lgbm][:200]
    z_hsj = art_hopskipjump(lab.lgbm.predict, small, max_iter=10, max_eval=300)
    hsj_evaded = lab.lgbm.predict(z_hsj) < 0.5
    hsj_valid = validate(z_hsj, lab.pre).valid
    report["hopskipjump_lgbm"] = {
        "rows": len(small),
        "evasion_rate": float(hsj_evaded.mean()),
        "valid_rate": float(hsj_valid.mean()),
        "evaded_and_valid": float((hsj_evaded & hsj_valid).mean()),
        "median_l2": float(np.median(np.linalg.norm(z_hsj - small, axis=1))),
    }

    with mlflow_run("evasion", "baseline_mlp", {"rows": len(rows)}):
        for tm in THREAT_MODELS.values():
            for budget, values in report[f"constrained_{tm.name}"]["whitebox_pgd_mlp"].items():
                log_metrics(f"{tm.name}_b{budget}_", values)
        for name, values in paper.items():
            log_metrics(f"{name}_", values)
    report["importance"] = importance(lab)
    save_json("m3_evasion", report)
    return {k: v for k, v in report.items() if k != "paper_attacks"}
