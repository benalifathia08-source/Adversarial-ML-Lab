"""Input validation: is this feature vector something a real network flow could produce?

Cheap, model-independent checks on the preprocessed vector z. They do not stop a careful attacker (the
constrained attack passes them by design), but they reject most "paper" adversarial examples, which change
features in ways no flow can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from advml.data import Preprocessor


@dataclass
class ValidationReport:
    valid: np.ndarray
    reasons: dict[str, np.ndarray]

    def summary(self) -> dict[str, float]:
        out = {"valid_rate": float(self.valid.mean()) if len(self.valid) else 0.0}
        out |= {f"fail_{k}": float(v.mean()) for k, v in self.reasons.items()}
        return out


def invert_numeric(z: np.ndarray, pre: Preprocessor) -> np.ndarray:
    n = len(pre.numeric)
    logged = z[:, :n].astype(np.float64) * np.asarray(pre.std) + np.asarray(pre.mean)
    return np.expm1(logged)


def validate(z: np.ndarray, pre: Preprocessor, integer_cols: list[str] | None = None) -> ValidationReport:
    n = len(pre.numeric)
    raw = invert_numeric(z, pre)
    col = {name: i for i, name in enumerate(pre.numeric)}
    reasons: dict[str, np.ndarray] = {}

    reasons["negative"] = np.any(raw < -1e-3, axis=1)

    # one-hot blocks: every value 0 or 1, exactly one 1 per block
    onehot = z[:, n:]
    bad = np.any(np.minimum(np.abs(onehot), np.abs(onehot - 1)) > 0.01, axis=1)
    start = 0
    for values in pre.categories.values():
        width = len(values) + 1
        bad |= np.abs(onehot[:, start : start + width].sum(axis=1) - 1) > 0.01
        start += width
    reasons["onehot"] = bad

    # counters and sizes are integers (checked where float precision allows it)
    integer_cols = integer_cols or [c for c in ("spkts", "dpkts", "sbytes", "dbytes", "smean", "dmean",
                                                "trans_depth") if c in col]
    ints = raw[:, [col[c] for c in integer_cols]]
    small = np.abs(ints) < 1e4
    reasons["integer"] = np.any(small & (np.abs(ints - np.round(ints)) > 0.02), axis=1)

    # arithmetic between features: mean packet size = bytes / packets
    def mean_mismatch(bytes_col: str, pkts_col: str, mean_col: str) -> np.ndarray:
        pkts = raw[:, col[pkts_col]]
        expected = np.floor(raw[:, col[bytes_col]] / np.maximum(np.round(pkts), 1) + 1e-6)
        mask = np.round(pkts) >= 1
        return mask & (np.abs(raw[:, col[mean_col]] - expected) > 1.5 + 1e-3 * expected)

    reasons["smean"] = mean_mismatch("sbytes", "spkts", "smean")
    reasons["dmean"] = mean_mismatch("dbytes", "dpkts", "dmean")

    valid = ~np.any(np.stack(list(reasons.values()), axis=1), axis=1)
    return ValidationReport(valid, reasons)
