"""Score network flows with the released model: python -m advml.score [--csv FILE | --sample N] [--attack]

The model is verified (signature, hashes, safe formats) before it is loaded. With --attack, each attack flow is
also given the strongest realistic evasion from this lab (attacker owns both ends, may add up to +100% traffic,
fakes replies) and scored again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from advml import config
from advml.data import Preprocessor, Split, load_raw, to_split
from advml.models import MLP, predict_proba


def score_frame(df: pd.DataFrame, model: MLP, pre: Preprocessor) -> tuple[Split, np.ndarray]:
    """P(attack) for every row of a UNSW-NB15-format table (label / attack_cat columns are optional)."""
    df = df.copy()
    if config.LABEL not in df:
        df[config.LABEL] = -1
    if config.CATEGORY not in df:
        df[config.CATEGORY] = "?"
    missing = [c for c in pre.numeric + config.CATEGORICAL if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    split = to_split(df, pre)
    return split, predict_proba(model, split.features(pre))


def attack_and_rescore(split: Split, model: MLP, pre: Preprocessor, budget: float = 1.0) -> np.ndarray:
    """Apply the both-ends constrained attack to every row and return the new P(attack)."""
    from advml.attacks import BOTH_ENDS, constrained_pgd
    from advml.models import RawPipeline

    res = constrained_pgd(RawPipeline(model, pre), split, budget, tm=BOTH_ENDS, steps=60, restarts=3)
    z = np.concatenate([pre.transform_numeric(res.num_raw_adv), split.cat], axis=1)
    return predict_proba(model, z)


def _sample(n: int, seed: int) -> pd.DataFrame:
    """n attack and n normal flows from the official training set (downloaded and hash-checked if needed)."""
    from advml.data import ensure_dataset

    ensure_dataset()
    df = load_raw("UNSW_NB15_training-set.csv")
    rng = np.random.default_rng(seed)
    parts = [df[df[config.LABEL] == lab].iloc[rng.choice((df[config.LABEL] == lab).sum(), n, replace=False)]
             for lab in (1, 0)]
    return pd.concat(parts, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m advml.score")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="flows in UNSW-NB15 format")
    source.add_argument("--sample", type=int, default=4, help="N attack + N normal flows from UNSW-NB15 (default 4)")
    parser.add_argument("--attack", action="store_true", help="also try the realistic evasion attack on attack flows")
    parser.add_argument("--model", type=Path, default=config.MODELS / "release")
    parser.add_argument("--key", type=Path, default=config.KEYS / "release.public.pem")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    from advml.supply_chain import VerificationError, load_public, load_verified

    try:
        model, pre = load_verified(args.model, load_public(args.key))
    except VerificationError as exc:
        print(f"[REFUSED] model not loaded: {exc}")
        return 1
    shown = args.model.resolve()
    shown = shown.relative_to(config.ROOT) if shown.is_relative_to(config.ROOT) else shown
    print(f"[ok] model verified (Ed25519 signature, hashes, safe formats): {shown.as_posix()}")
    df = pd.read_csv(args.csv, encoding="utf-8-sig") if args.csv else _sample(args.sample, args.seed)
    split, proba = score_frame(df, model, pre)
    after = None
    attack_rows = np.where(split.y == 1)[0]
    if args.attack and len(attack_rows):
        after = np.full(len(split), np.nan)
        after[attack_rows] = attack_and_rescore(split.subset(attack_rows), model, pre)

    head = f"{'#':>3}  {'truth':<16} {'P(attack)':>9}  {'verdict':<8}"
    if after is not None:
        head += f"  {'after evasion attempt (+100% traffic, fake replies)':<52}"
    print(head)
    print("-" * len(head))
    for i in range(len(split)):
        truth = "normal" if split.y[i] == 0 else (split.attack_cat[i] if split.y[i] == 1 else "?")
        line = f"{i:>3}  {truth:<16} {proba[i]:>9.3f}  {'ATTACK' if proba[i] >= 0.5 else 'normal':<8}"
        if after is not None and not np.isnan(after[i]):
            verdict = "still ATTACK" if after[i] >= 0.5 else "EVADED"
            line += f"  {after[i]:>9.3f}  {verdict}"
        print(line)
    if split.y.min() >= 0:
        correct = ((proba >= 0.5) == (split.y == 1)).mean()
        print(f"\n{correct:.0%} of {len(split)} flows classified correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
