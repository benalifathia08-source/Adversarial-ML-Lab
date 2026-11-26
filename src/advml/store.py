"""Save and load models without pickle: weights in safetensors, everything else in JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file

from advml.data import Preprocessor
from advml.models import MLP, MaskedMLP

WEIGHTS = "model.safetensors"
ARCH = "architecture.json"
PREPROCESSOR = "preprocessor.json"


def save_model(model: MLP, pre: Preprocessor, directory: Path, extra: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(directory / WEIGHTS))
    arch = {"in_dim": model.in_dim, "hidden": list(model.hidden), "masked": isinstance(model, MaskedMLP),
            **(extra or {})}
    (directory / ARCH).write_text(json.dumps(arch, indent=1), encoding="utf-8")
    pre.to_json(directory / PREPROCESSOR)
    return directory


def load_model(directory: Path) -> tuple[MLP, Preprocessor]:
    """Load weights and config. No code from the files is ever executed (safetensors + JSON only)."""
    arch = json.loads((directory / ARCH).read_text(encoding="utf-8"))
    if arch.get("masked"):  # the feature mask is a tensor inside the safetensors file
        model = MaskedMLP(arch["in_dim"], np.ones(arch["in_dim"], dtype=np.float32), tuple(arch["hidden"]))
    else:
        model = MLP(arch["in_dim"], tuple(arch["hidden"]))
    model.load_state_dict(load_file(str(directory / WEIGHTS)))
    model.eval()
    return model, Preprocessor.from_json(directory / PREPROCESSOR)
