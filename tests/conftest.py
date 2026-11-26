"""Synthetic UNSW-NB15-shaped flows: same columns and the same arithmetic between features.

Lets the whole test suite (and CI) run in seconds without downloading the dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from advml.data import Preprocessor, Split, to_split

NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]


def make_flows(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    spkts = rng.integers(1, 40, n)
    # attacks often get no reply; normal flows usually do (this is the signal the model learns)
    has_reply = np.where(label == 1, rng.random(n) < 0.3, rng.random(n) < 0.9)
    dpkts = np.where(has_reply, rng.integers(1, 40, n), 0)
    smean = rng.integers(28, 900, n)
    dmean = np.where(dpkts > 0, rng.integers(28, 1400, n), 0)
    sbytes = smean * spkts + rng.integers(0, 1, n)
    dbytes = dmean * dpkts
    dur = rng.uniform(0.001, 5.0, n)
    df = pd.DataFrame({c: rng.integers(0, 5, n).astype(float) for c in NUMERIC})
    df["dur"] = dur
    df["spkts"], df["dpkts"] = spkts, dpkts
    df["sbytes"], df["dbytes"] = sbytes, dbytes
    df["smean"] = np.floor(sbytes / spkts)
    df["dmean"] = np.where(dpkts > 0, np.floor(dbytes / np.maximum(dpkts, 1)), 0)
    df["rate"] = (spkts + dpkts - 1) / dur
    df["sload"] = 0.75 * sbytes * 8 / dur
    df["dload"] = 0.9 * dbytes * 8 / dur
    df["sinpkt"] = np.where(spkts > 1, dur * 1000 / np.maximum(spkts - 1, 1), 0)
    df["dinpkt"] = np.where(dpkts > 1, dur * 1000 / np.maximum(dpkts - 1, 1), 0)
    df["sjit"] = rng.uniform(0, 50, n)
    df["djit"] = np.where(dpkts > 1, rng.uniform(0, 50, n), 0)
    df["proto"] = rng.choice(["tcp", "udp", "unas"], n)
    df["service"] = rng.choice(["-", "dns", "http"], n)
    df["state"] = rng.choice(["FIN", "INT", "CON"], n)
    df["attack_cat"] = np.where(label == 1, rng.choice(["Exploits", "Generic", "Fuzzers"], n), "Normal")
    df["label"] = label
    df["id"] = np.arange(n)
    return df


@pytest.fixture(scope="session")
def flows() -> pd.DataFrame:
    return make_flows()


@pytest.fixture(scope="session")
def pre(flows) -> Preprocessor:
    return Preprocessor.fit(flows, min_count=10)


@pytest.fixture(scope="session")
def split(flows, pre) -> Split:
    return to_split(flows, pre)


@pytest.fixture(scope="session")
def trained(split, pre):
    from advml.models import TrainConfig, train_mlp

    return train_mlp(split, pre, TrainConfig(epochs=4, hidden=(32, 16)))
