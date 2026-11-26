"""Verify a model folder before using it: python -m advml.verify <folder> [--gate]

Checks the Ed25519 signature, every file hash, safe formats only, runs ModelScan, and (with --gate) applies
the promotion policy. Exit code 0 = OK, 1 = refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from advml import config
from advml.supply_chain import VerificationError, gate, load_public, scan, scan_issues, verify_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m advml.verify")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--key", type=Path, default=config.KEYS / "release.public.pem")
    parser.add_argument("--gate", action="store_true", help="also apply the promotion policy")
    args = parser.parse_args(argv)
    key = load_public(args.key)
    print(f"model folder : {args.folder}")
    print(f"public key   : {args.key.name}")
    try:
        manifest = verify_model(args.folder, key)
    except VerificationError as exc:
        print(f"[REFUSED]    integrity: {exc}")
        return 1
    print(f"[ok]         signature valid (Ed25519), signed {manifest['signed_at']}")
    for name, digest in manifest["files"].items():
        print(f"[ok]         sha256 {digest[:16]}…  {name}")
    issues = scan_issues(scan(args.folder))
    print(f"[{'ok' if issues == 0 else 'REFUSED'}]{' ' * (9 if issues == 0 else 4)}ModelScan: {issues} issue(s)")
    if issues:
        return 1
    metrics = manifest["metadata"].get("metrics", {})
    for k, v in metrics.items():
        print(f"             {k:28s} {v:.4f}")
    if args.gate:
        decision = gate(args.folder, key)
        if decision.accepted:
            print("[ACCEPTED]   gate: may be promoted to 'champion'")
        for reason in decision.reasons:
            print(f"[REJECTED]   gate: {reason}")
        return 0 if decision.accepted else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
