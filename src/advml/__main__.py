"""Command line: python -m advml <stage>."""

from __future__ import annotations

import argparse
import importlib
import json

STAGES = ["baseline", "evasion", "poisoning", "defenses", "supply_chain", "final"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="advml")
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    module = importlib.import_module(f"advml.stages.{args.stage}")
    result = module.run()
    print(json.dumps(result, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
