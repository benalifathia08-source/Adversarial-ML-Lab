"""M1 + M2: data report and baseline models (validation split only)."""

from __future__ import annotations

import time

import numpy as np

from advml import config
from advml.data import dev_splits
from advml.models import TrainConfig, metrics, per_category_recall, predict_proba, train_lgbm, train_mlp
from advml.results import log_metrics, mlflow_run, save_json
from advml.store import save_model


def run() -> dict:
    report: dict = {"data": {}, "models": {}}
    for use_ttl in (True, False):
        tag = "with_ttl" if use_ttl else "no_ttl"
        pre, train, val = dev_splits(use_ttl=use_ttl)
        if not use_ttl:
            report["data"] = {
                "train_rows": len(train), "val_rows": len(val),
                "train_attack_share": float(train.y.mean()), "val_attack_share": float(val.y.mean()),
                "numeric_features": len(pre.numeric), "onehot_features": len(pre.onehot_columns),
                "val_by_category": {c: int(n) for c, n in zip(*np.unique(val.attack_cat, return_counts=True),
                                                               strict=True)},
            }
        z_val = val.features(pre)
        cfg = TrainConfig()
        with mlflow_run("baseline", f"mlp_{tag}", {"model": "mlp", "use_ttl": use_ttl, **cfg.__dict__}):
            t0 = time.time()
            mlp = train_mlp(train, pre, cfg)
            seconds = time.time() - t0
            p_mlp = predict_proba(mlp, z_val)
            m_mlp = metrics(val.y, p_mlp) | {"train_seconds": seconds}
            log_metrics("val_", m_mlp)
        with mlflow_run("baseline", f"lgbm_{tag}", {"model": "lightgbm", "use_ttl": use_ttl}):
            t0 = time.time()
            lgbm = train_lgbm(train, pre)
            seconds = time.time() - t0
            p_lgbm = lgbm.predict_proba(z_val)[:, 1]
            m_lgbm = metrics(val.y, p_lgbm) | {"train_seconds": seconds}
            log_metrics("val_", m_lgbm)
        report["models"][f"mlp_{tag}"] = m_mlp | {"per_category_recall": per_category_recall(
            val.attack_cat, val.y, p_mlp)}
        report["models"][f"lgbm_{tag}"] = m_lgbm | {"per_category_recall": per_category_recall(
            val.attack_cat, val.y, p_lgbm)}
        if not use_ttl:
            save_model(mlp, pre, config.WORK / "baseline_mlp")
            lgbm.booster_.save_model(str(config.WORK / "baseline_lgbm.txt"))
    save_json("m2_baseline", report)
    return report
