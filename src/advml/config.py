"""Paths, seeds and the feature groups that define the threat model."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ADVML_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
WORK = ARTIFACTS / "work"  # local, not committed
RESULTS = ROOT / "results"  # committed: metrics JSON and figures
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"  # committed: the released, signed model
KEYS = ROOT / "keys"

SEED = 1337

# Official UNSW-NB15 partition files (the Hugging Face mirror swaps their names; see data.py).
DATASET_SOURCE = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/3eb02c4a0a29866b7abcb5ef77c45cf4fcc8f6b0"
DATASET_FILES = {
    # local official name: (file name on the mirror, sha256, expected rows)
    "UNSW_NB15_training-set.csv": (
        "test.csv",
        "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
        175_341,
    ),
    "UNSW_NB15_testing-set.csv": (
        "train.csv",
        "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
        82_332,
    ),
}

LABEL = "label"
CATEGORY = "attack_cat"
DROP = ["id"]
CATEGORICAL = ["proto", "service", "state"]

# Testbed artefacts: TTL values separate the classes almost perfectly in UNSW-NB15 because of how the
# lab was wired, and an attacker can set TTL freely. The main model does not use them.
TTL_FEATURES = ["sttl", "dttl", "ct_state_ttl"]

# Threat model. The attacker controls the traffic they send, not the sensor, the victim's replies or the
# connection counters the sensor computes. They can only ADD to their traffic (padding, extra packets,
# delays), because the attack must still work.
ATTACKER_PRIMITIVES = ["dur", "spkts", "sbytes", "sjit"]
# Features derived from the primitives; recomputed from them so the example stays consistent.
ATTACKER_DERIVED = ["smean", "sload", "sinpkt", "rate"]
ATTACKER_CONTROLLED = ATTACKER_PRIMITIVES + ATTACKER_DERIVED
INTEGER_FEATURES = ["spkts"]
