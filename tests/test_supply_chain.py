import shutil

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from advml.store import save_model
from advml.supply_chain import (
    GatePolicy,
    VerificationError,
    gate,
    load_verified,
    make_malicious_checkpoint,
    scan,
    scan_issues,
    sign_model,
    verify_model,
)

GOOD = {"val_recall": 0.97, "val_fpr": 0.10, "evasion_both_ends_b1.0": 0.05}


@pytest.fixture
def keys():
    private = Ed25519PrivateKey.generate()
    return private, private.public_key()


@pytest.fixture
def signed(trained, pre, tmp_path, keys):
    directory = save_model(trained, pre, tmp_path / "model")
    sign_model(directory, keys[0], {"metrics": GOOD})
    return directory


def test_signed_model_loads(signed, keys):
    model, _ = load_verified(signed, keys[1])
    assert model is not None


def test_bit_flip_in_weights_is_refused(signed, keys):
    path = signed / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[-5] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(VerificationError, match="hash mismatch"):
        load_verified(signed, keys[1])


def test_extra_file_is_refused(signed, keys):
    (signed / "notes.json").write_text("{}")
    with pytest.raises(VerificationError, match="not covered"):
        verify_model(signed, keys[1])


def test_wrong_key_is_refused(signed):
    with pytest.raises(VerificationError, match="signature"):
        verify_model(signed, Ed25519PrivateKey.generate().public_key())


def test_edited_manifest_is_refused(signed, keys):
    manifest = signed / "manifest.json"
    manifest.write_text(manifest.read_text().replace("0.97", "0.99"))
    with pytest.raises(VerificationError, match="signature"):
        verify_model(signed, keys[1])


def test_unsigned_model_is_refused(signed, keys):
    (signed / "manifest.sig").unlink()
    with pytest.raises(VerificationError, match="not signed"):
        verify_model(signed, keys[1])


def test_signed_pickle_is_still_refused(signed, keys, trained, tmp_path):
    """Even a correctly signed model may not contain unsafe formats."""
    make_malicious_checkpoint(signed / "extra.pt", tmp_path / "marker.txt", trained.state_dict())
    sign_model(signed, keys[0], {"metrics": GOOD})
    with pytest.raises(VerificationError, match="unsafe file formats"):
        verify_model(signed, keys[1])


def test_malicious_checkpoint_runs_code_only_with_unsafe_loading(trained, tmp_path):
    marker = tmp_path / "marker.txt"
    evil = make_malicious_checkpoint(tmp_path / "evil.pt", marker, trained.state_dict())
    with pytest.raises(Exception):  # noqa: B017 - torch raises an UnpicklingError subclass
        torch.load(evil, weights_only=True)
    assert not marker.exists()
    torch.load(evil, weights_only=False)  # noqa: S614 - deliberate: proves the risk
    assert marker.exists()


def test_modelscan_flags_the_malicious_checkpoint_and_not_the_safe_model(trained, tmp_path, signed):
    evil = make_malicious_checkpoint(tmp_path / "evil.pt", tmp_path / "m.txt", trained.state_dict())
    assert scan_issues(scan(evil)) >= 1
    assert scan_issues(scan(signed)) == 0


def test_gate_accepts_a_good_model(signed, keys):
    assert gate(signed, keys[1]).accepted


@pytest.mark.parametrize(
    ("metrics", "reason"),
    [
        ({**GOOD, "evasion_both_ends_b1.0": 0.6}, "robustness"),
        ({**GOOD, "val_recall": 0.80}, "recall"),
        ({**GOOD, "val_fpr": 0.40}, "false-positive"),
    ],
)
def test_gate_rejects_weak_models(signed, keys, metrics, reason):
    sign_model(signed, keys[0], {"metrics": metrics})
    decision = gate(signed, keys[1], GatePolicy())
    assert not decision.accepted and any(reason in r for r in decision.reasons)


def test_gate_rejects_tampered_model(signed, keys, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(signed, copy)
    (copy / "architecture.json").write_text('{"in_dim": 1, "hidden": [1]}')
    decision = gate(copy, keys[1])
    assert not decision.accepted and decision.reasons[0].startswith("integrity")
