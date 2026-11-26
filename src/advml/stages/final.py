"""M7: the one and only run on the official UNSW-NB15 testing set.

Everything was tuned on the validation split; this stage reports how the baseline and the released
(hardened, signed) model behave on data neither has seen. It refuses to run twice unless forced, so the
test set stays a real test.
"""

from __future__ import annotations

import os

import numpy as np

from advml import config
from advml.attacks import BOTH_ENDS, ONE_SIDED, constrained_pgd
from advml.config import RESULTS
from advml.data import frozen_test
from advml.models import metrics, per_category_recall, predict_proba
from advml.results import log_metrics, mlflow_run, save_json
from advml.stages.common import pipeline
from advml.store import load_model
from advml.supply_chain import load_public, load_verified

N_ROWS = 2000
BUDGETS = [0.25, 1.0]


def run() -> dict:
    out_file = RESULTS / "m7_final_test.json"
    if out_file.exists() and os.environ.get("ADVML_FORCE_FINAL") != "1":
        raise SystemExit("The test set was already used (results/m7_final_test.json). Set ADVML_FORCE_FINAL=1 "
                         "only if you know why you are running it again.")
    baseline, pre = load_model(config.WORK / "baseline_mlp")
    released, released_pre = load_verified(config.MODELS / "release", load_public(config.KEYS / "release.public.pem"))
    assert released_pre.numeric == pre.numeric
    test = frozen_test(pre)
    z = test.features(pre)
    report: dict = {"rows": len(test), "attack_share": float(test.y.mean())}
    rng = np.random.default_rng(config.SEED)
    adversarial, _ = load_model(config.WORK / "hardened_mlp")
    for name, model in (("baseline", baseline), ("adversarial_training", adversarial), ("released", released)):
        proba = predict_proba(model, z)
        entry = {"clean": metrics(test.y, proba), "per_category_recall": per_category_recall(
            test.attack_cat, test.y, proba)}
        detected = np.where((test.y == 1) & (proba >= 0.5))[0]
        rows = test.subset(np.sort(rng.choice(detected, size=min(N_ROWS, len(detected)), replace=False)))
        pipe = pipeline(model, pre)
        for tm in (ONE_SIDED, BOTH_ENDS):
            entry[f"whitebox_{tm.name}"] = {
                str(b): constrained_pgd(pipe, rows, b, tm=tm, steps=60, restarts=3).evasion_rate for b in BUDGETS}
        report[name] = entry
    with mlflow_run("final_test", "official_testing_set"):
        for name in ("baseline", "adversarial_training", "released"):
            log_metrics(f"{name}_test_", report[name]["clean"])
    save_json("m7_final_test", report)
    return report
