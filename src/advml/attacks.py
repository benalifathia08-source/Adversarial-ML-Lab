"""Evasion attacks.

Two families:

* "Paper" attacks (ART FGSM / PGD / HopSkipJump) that change any feature of the preprocessed vector by a
  small amount. They are the standard benchmark, but their outputs are mostly not real traffic: they change
  sensor-computed counters, the protocol one-hot, or break the arithmetic between features.
* Constrained attacks written for a threat model: they only ADD traffic the attacker controls (padding,
  extra packets, delays), recompute every derived feature so the flow stays consistent, respect the Ethernet
  MTU, and round packet and byte counts to integers before the result is judged.

Threat models:
* ``ONE_SIDED``: the attacker controls only what they send (an attack on someone else's server).
* ``BOTH_ENDS``: the attacker controls both endpoints (C2 or exfiltration to a server they own), so the
  replies can be padded and delayed too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from advml import config
from advml.data import Preprocessor, Split
from advml.models import MLP, RawPipeline

ALL_PRIMITIVES = ["dur", "spkts", "sbytes", "sjit", "dpkts", "dbytes", "djit"]
DERIVED = ["smean", "dmean", "sload", "dload", "sinpkt", "dinpkt", "rate"]


@dataclass(frozen=True)
class ThreatModel:
    name: str
    primitives: tuple[str, ...]

    @property
    def controlled(self) -> list[str]:
        """Every feature the attack may change (primitives plus what is derived from them).

        Loads and inter-packet times of BOTH directions depend on the duration, so even a one-sided attacker
        who adds delay changes the reply-side rates the sensor computes.
        """
        derived = ["smean", "sload", "sinpkt", "rate", "dload", "dinpkt"]
        if {"dpkts", "dbytes"} & set(self.primitives):
            derived.append("dmean")
        return list(self.primitives) + derived


ONE_SIDED = ThreatModel("one_sided", ("dur", "spkts", "sbytes", "sjit"))
BOTH_ENDS = ThreatModel("both_ends", tuple(ALL_PRIMITIVES))
THREAT_MODELS = {t.name: t for t in (ONE_SIDED, BOTH_ENDS)}


class Traffic:
    """Column positions of the traffic features inside preprocessor.numeric."""

    def __init__(self, pre: Preprocessor):
        self.col = {n: pre.numeric.index(n) for n in ALL_PRIMITIVES + DERIVED}


# Fitted on the UNSW-NB15 training set (median ratios), used only when a direction had no traffic before.
K_LOAD = {"s": 0.75, "d": 0.90}  # load ~ k * bytes * 8 / dur
K_INPKT = {"s": 1.00, "d": 0.937}  # inpkt ~ k * dur * 1000 / (pkts - 1)
MIN_PKT_BYTES = 28.0  # smallest mean packet size seen in the data
MTU_BYTES = 1504.0  # largest mean packet size seen in the data (Ethernet MTU + framing)


def _ref(num_raw: torch.Tensor, tm: ThreatModel, tr: Traffic) -> torch.Tensor:
    """What "budget b" is relative to, per primitive: its own value, or the request side if it was zero.

    The attacker may ADD up to b * ref. With ref = value this is the same as "grow by up to (1 + b) times".
    Replies that did not exist are measured against the request (e.g. up to b * spkts reply packets).
    """
    c = tr.col
    refs = []
    for p in tm.primitives:
        base = num_raw[:, c[p]]
        if p == "dpkts":
            base = torch.maximum(base, num_raw[:, c["spkts"]])
        elif p == "dbytes":
            base = torch.maximum(base, num_raw[:, c["sbytes"]])
        elif p == "dur":
            base = torch.clamp(base, min=1e-3)
        refs.append(base)
    return torch.stack(refs, dim=1)


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a / torch.clamp(b, min=1e-9)


def apply_changes(num_raw: torch.Tensor, v: torch.Tensor, tm: ThreatModel, tr: Traffic,
                  ref: torch.Tensor) -> torch.Tensor:
    """Add v * ref to the threat model's primitives (v >= 0), then recompute every derived feature."""
    c = tr.col
    base = {name: num_raw[:, i] for name, i in c.items()}
    new = dict(base)
    for k, p in enumerate(tm.primitives):
        new[p] = base[p] + v[:, k] * ref[:, k]
    dur_b, dur_n = base["dur"], new["dur"]
    for side in ("s", "d"):
        pk_b, pk_n = base[f"{side}pkts"], new[f"{side}pkts"]
        by_b, by_n = base[f"{side}bytes"], new[f"{side}bytes"]
        new[f"{side}mean"] = torch.where(pk_n > 0, _safe_div(by_n, pk_n), base[f"{side}mean"])
        scaled = base[f"{side}load"] * _safe_div(by_n, by_b) * _safe_div(dur_b, dur_n)
        fresh = K_LOAD[side] * by_n * 8 / torch.clamp(dur_n, min=1e-6)
        new[f"{side}load"] = torch.where((by_b > 0) & (dur_b > 0), scaled, fresh)
        scaled = base[f"{side}inpkt"] * _safe_div(dur_n, dur_b) * _safe_div(pk_b - 1, pk_n - 1)
        fresh = K_INPKT[side] * dur_n * 1000 * _safe_div(torch.ones_like(pk_n), pk_n - 1)
        new[f"{side}inpkt"] = torch.where(
            (pk_b >= 2) & (dur_b > 0), scaled, torch.where(pk_n >= 2, fresh, base[f"{side}inpkt"]))
    # UNSW-NB15: rate = (spkts + dpkts - 1) / dur (checked: holds for 99.8% of training rows)
    tot_b = base["spkts"] + base["dpkts"] - 1
    tot_n = new["spkts"] + new["dpkts"] - 1
    scaled = base["rate"] * _safe_div(tot_n, tot_b) * _safe_div(dur_b, dur_n)
    fresh = torch.clamp(tot_n, min=0) / torch.clamp(dur_n, min=1e-6)
    new["rate"] = torch.where((base["rate"] > 0) & (tot_b > 0) & (dur_b > 0), scaled, fresh)
    columns = list(num_raw.unbind(dim=1))
    for name, i in c.items():
        columns[i] = new[name]
    return torch.stack(columns, dim=1)


def _project(v: torch.Tensor, num_raw: torch.Tensor, tm: ThreatModel, tr: Traffic, ref: torch.Tensor,
             budget: float) -> torch.Tensor:
    """Keep v in [0, budget] and keep packets and bytes physically possible.

    Every packet carries between MIN_PKT_BYTES and MTU_BYTES, so new packets need new bytes, and padding
    cannot push the mean packet size over the MTU.
    """
    v = torch.clamp(v, 0.0, budget)
    pos = {p: k for k, p in enumerate(tm.primitives)}
    cols = list(v.unbind(dim=1))
    c = tr.col
    for side in ("s", "d"):
        kb, kp = pos.get(f"{side}bytes"), pos.get(f"{side}pkts")
        by_b, pk_b = num_raw[:, c[f"{side}bytes"]], num_raw[:, c[f"{side}pkts"]]
        by_max = by_b + (budget * ref[:, kb] if kb is not None else 0)
        if kp is not None:  # no more packets than the byte budget can fill
            pk_max = torch.maximum(pk_b, torch.floor(by_max / MIN_PKT_BYTES))
            cols[kp] = torch.minimum(cols[kp], _safe_div(pk_max - pk_b, ref[:, kp]))
            pk_n = pk_b + cols[kp] * ref[:, kp]
        else:
            pk_n = pk_b
        if kb is not None:
            lo = _safe_div(torch.clamp(MIN_PKT_BYTES * pk_n - by_b, min=0), ref[:, kb])
            hi = _safe_div(torch.clamp(MTU_BYTES * pk_n - by_b, min=0), ref[:, kb])
            cols[kb] = torch.clamp(torch.minimum(torch.maximum(cols[kb], lo), hi), 0.0, budget)
    return torch.stack(cols, dim=1)


def _finalize(num_raw: torch.Tensor, v: torch.Tensor, tm: ThreatModel, tr: Traffic,
              ref: torch.Tensor) -> torch.Tensor:
    """Round packet counts up to integers, recompute from them, then fix bytes and mean sizes exactly."""
    v = v.clone()
    for k, p in enumerate(tm.primitives):
        if p in ("spkts", "dpkts"):
            base = num_raw[:, tr.col[p]]
            rounded = torch.ceil(base + v[:, k] * ref[:, k] - 1e-6)
            v[:, k] = _safe_div(rounded - base, ref[:, k])
    out = apply_changes(num_raw, v, tm, tr, ref)
    for side in ("s", "d"):
        b, p, m = (tr.col[f"{side}{x}"] for x in ("bytes", "pkts", "mean"))
        out[:, b] = torch.maximum(torch.ceil(out[:, b] - 1e-6), MIN_PKT_BYTES * out[:, p])
        exact = torch.floor(out[:, b] / torch.clamp(out[:, p], min=1))
        out[:, m] = torch.where(out[:, p] > 0, exact, out[:, m])
    return out


@dataclass
class EvasionResult:
    num_raw_adv: np.ndarray
    evaded: np.ndarray  # bool per row: classified as normal after the attack
    bytes_added_ratio: np.ndarray  # total bytes (both directions) added, relative
    duration_added_ratio: np.ndarray

    @property
    def evasion_rate(self) -> float:
        return float(self.evaded.mean()) if len(self.evaded) else 0.0


def constrained_pgd(
    pipeline: RawPipeline,
    rows: Split,
    budget: float,
    tm: ThreatModel = ONE_SIDED,
    steps: int = 40,
    restarts: int = 2,
    seed: int = config.SEED,
) -> EvasionResult:
    """White-box PGD in "how much traffic do I add" space.

    budget = 0.5 means each primitive may grow by up to 50% of its reference (never shrink).
    """
    tr = Traffic(pipeline.pre)
    num_raw = torch.from_numpy(rows.num_raw.astype(np.float64))
    cat = torch.from_numpy(rows.cat)
    ref = _ref(num_raw, tm, tr)
    width = len(tm.primitives)
    target = torch.zeros(len(rows), dtype=torch.long)  # class 0 = normal
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    gen = torch.Generator().manual_seed(seed)
    best_v = torch.zeros(len(rows), width, dtype=torch.float64)
    best_ok = torch.zeros(len(rows), dtype=torch.bool)
    step = 2.5 * budget / steps
    pipeline.eval()
    for restart in range(restarts):
        v = torch.zeros(len(rows), width, dtype=torch.float64)
        if restart > 0:
            v = torch.rand(len(rows), width, generator=gen, dtype=torch.float64) * budget
        v = _project(v, num_raw, tm, tr, ref, budget)
        for _ in range(steps):
            v.requires_grad_(True)
            loss = loss_fn(pipeline(apply_changes(num_raw, v, tm, tr, ref), cat), target)
            (grad,) = torch.autograd.grad(loss, v)
            with torch.no_grad():
                v = _project(v - step * torch.sign(torch.nan_to_num(grad)), num_raw, tm, tr, ref, budget)
        with torch.no_grad():
            ok = pipeline(_finalize(num_raw, v, tm, tr, ref), cat).argmax(dim=1) == 0
            newly = ok & ~best_ok
            best_v[newly] = v[newly]
            if restart == 0:
                best_v[~ok] = v[~ok]
            best_ok |= ok
    with torch.no_grad():
        adv = _finalize(num_raw, best_v, tm, tr, ref)
        evaded = pipeline(adv, cat).argmax(dim=1) == 0
    return _result(rows.num_raw, adv.numpy(), evaded.numpy(), tr)


PredictAttack = Callable[[np.ndarray, np.ndarray], np.ndarray]


def random_increase(
    predict_attack: PredictAttack, rows: Split, pre: Preprocessor, budget: float,
    tm: ThreatModel = ONE_SIDED, queries: int = 50, seed: int = config.SEED,
) -> EvasionResult:
    """Black-box, decision-only attack: try random allowed additions, keep the first that evades.

    predict_attack(num_raw, cat) -> bool array (True = flagged). Works against any model, including trees.
    """
    tr = Traffic(pre)
    rng = np.random.default_rng(seed)
    num_raw = torch.from_numpy(rows.num_raw.astype(np.float64))
    ref_all = _ref(num_raw, tm, tr)
    best = num_raw.clone()
    evaded = np.zeros(len(rows), dtype=bool)
    for _ in range(queries):
        todo = np.where(~evaded)[0]
        if not len(todo):
            break
        v = torch.from_numpy(rng.uniform(0, budget, size=(len(todo), len(tm.primitives))))
        sub, ref = num_raw[todo], ref_all[todo]
        cand = _finalize(sub, _project(v, sub, tm, tr, ref, budget), tm, tr, ref)
        flagged = predict_attack(cand.numpy(), rows.cat[todo])
        hit = todo[~flagged]
        best[hit] = cand[~flagged]
        evaded[hit] = True
    return _result(rows.num_raw, best.numpy(), evaded, tr)


def _result(before: np.ndarray, after: np.ndarray, evaded: np.ndarray, tr: Traffic) -> EvasionResult:
    c = tr.col
    total_before = before[:, c["sbytes"]] + before[:, c["dbytes"]]
    total_after = after[:, c["sbytes"]] + after[:, c["dbytes"]]
    dur_before = before[:, c["dur"]]
    return EvasionResult(
        after,
        np.asarray(evaded, dtype=bool),
        np.where(total_before > 0, total_after / np.maximum(total_before, 1e-9) - 1.0, 0.0),
        np.where(dur_before > 0, after[:, c["dur"]] / np.maximum(dur_before, 1e-9) - 1.0, 0.0),
    )


# ---------------------------------------------------------------- ART "paper" attacks
def art_classifier(model: MLP, in_dim: int):
    from art.estimators.classification import PyTorchClassifier

    return PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(), input_shape=(in_dim,), nb_classes=2,
        optimizer=None, device_type="cpu",
    )


def _to_normal(n: int) -> np.ndarray:
    return np.tile([1.0, 0.0], (n, 1)).astype(np.float32)


def art_fgsm(model: MLP, z: np.ndarray, eps: float) -> np.ndarray:
    from art.attacks.evasion import FastGradientMethod

    attack = FastGradientMethod(art_classifier(model, z.shape[1]), eps=eps, targeted=True)
    return attack.generate(x=z.astype(np.float32), y=_to_normal(len(z)))


def art_pgd(model: MLP, z: np.ndarray, eps: float, max_iter: int = 40) -> np.ndarray:
    from art.attacks.evasion import ProjectedGradientDescent

    attack = ProjectedGradientDescent(
        art_classifier(model, z.shape[1]), eps=eps, eps_step=eps / 8, max_iter=max_iter, targeted=True,
        num_random_init=0, verbose=False,
    )
    return attack.generate(x=z.astype(np.float32), y=_to_normal(len(z)))


def art_hopskipjump(predict_proba_fn: Callable, z: np.ndarray, max_iter: int = 20, max_eval: int = 500):
    """Decision-based black-box attack (ART HopSkipJump) against any model's predict function."""
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import BlackBoxClassifier

    def predict(x: np.ndarray) -> np.ndarray:
        p = predict_proba_fn(x.astype(np.float32))
        return np.stack([1 - p, p], axis=1)

    lo, hi = z.min(axis=0) - 1, z.max(axis=0) + 1
    clf = BlackBoxClassifier(predict, input_shape=(z.shape[1],), nb_classes=2, clip_values=(lo, hi))
    attack = HopSkipJump(clf, targeted=False, max_iter=max_iter, max_eval=max_eval, init_eval=50,
                         verbose=False)
    return attack.generate(x=z.astype(np.float32))
