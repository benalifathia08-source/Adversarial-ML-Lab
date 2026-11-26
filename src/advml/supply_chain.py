"""Model supply chain security.

* Unsafe formats: a pickle-based PyTorch checkpoint can run code when loaded. ``make_malicious_checkpoint``
  builds one with a harmless payload (it writes a marker file) to show the risk and to test the scanner.
* Scanning: ModelScan inspects model files for dangerous pickle opcodes and imports.
* Safe formats: the release is safetensors + JSON only, and loading never unpickles anything.
* Signing: an Ed25519-signed manifest lists the SHA-256 of every file; loading refuses a model whose
  signature or hashes do not match, or that contains files the manifest does not list.
* Registry gate: a model is promoted only if it is signed, scans clean, uses only safe formats, and meets
  the accuracy and robustness thresholds.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

MANIFEST = "manifest.json"
SIGNATURE = "manifest.sig"
SAFE_SUFFIXES = {".safetensors", ".json"}


# ---------------------------------------------------------------- unsafe format demo
class _Payload:
    """Pickled object whose unpickling calls exec(). Harmless here: it only writes a marker file."""

    def __init__(self, marker: Path):
        self.marker = str(marker)

    def __reduce__(self):
        code = f"open({self.marker!r}, 'w').write('code ran while loading the model')"
        return (exec, (code,))


def make_malicious_checkpoint(path: Path, marker: Path, state_dict: dict) -> Path:
    """A normal-looking PyTorch checkpoint (torch.save, pickle inside) carrying the payload."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "metadata": _Payload(marker)}, path)
    return path


def scan(path: Path) -> dict:
    """Run ModelScan on a file or directory; return its report (summary + issues)."""
    from modelscan.modelscan import ModelScan

    return ModelScan().scan(str(path))


def scan_issues(report: dict) -> int:
    return int(report.get("summary", {}).get("total_issues", 0))


# ---------------------------------------------------------------- keys and signing
def generate_keypair(private_path: Path, public_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
    public_path.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                          serialization.PublicFormat.SubjectPublicKeyInfo))


def load_private(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("expected an Ed25519 private key")
    return key


def load_public(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("expected an Ed25519 public key")
    return key


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_model(directory: Path, private_key: Ed25519PrivateKey, metadata: dict) -> dict:
    """Write manifest.json (file hashes + metadata) and manifest.sig (Ed25519 over the canonical JSON)."""
    files = {p.name: _sha256(p) for p in sorted(directory.iterdir())
             if p.is_file() and p.name not in (MANIFEST, SIGNATURE)}
    signed_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {"files": files, "metadata": metadata, "signed_at": signed_at}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    signature = private_key.sign(_canonical(manifest))
    (directory / SIGNATURE).write_text(base64.b64encode(signature).decode("ascii"), encoding="ascii")
    return manifest


class VerificationError(Exception):
    pass


def verify_model(directory: Path, public_key: Ed25519PublicKey) -> dict:
    """Check signature, every hash, no extra or missing files, and safe formats only. Returns the manifest."""
    manifest_path, sig_path = directory / MANIFEST, directory / SIGNATURE
    if not manifest_path.exists() or not sig_path.exists():
        raise VerificationError("model is not signed (manifest or signature missing)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        public_key.verify(base64.b64decode(sig_path.read_text(encoding="ascii")), _canonical(manifest))
    except (InvalidSignature, ValueError) as exc:
        raise VerificationError("signature does not match the manifest") from exc
    present = {p.name for p in directory.iterdir() if p.is_file() and p.name not in (MANIFEST, SIGNATURE)}
    listed = set(manifest["files"])
    if present - listed:
        raise VerificationError(f"files not covered by the signature: {sorted(present - listed)}")
    if listed - present:
        raise VerificationError(f"signed files missing: {sorted(listed - present)}")
    for name, digest in manifest["files"].items():
        if _sha256(directory / name) != digest:
            raise VerificationError(f"hash mismatch for {name} (file was changed after signing)")
    unsafe = [n for n in listed if Path(n).suffix not in SAFE_SUFFIXES]
    if unsafe:
        raise VerificationError(f"unsafe file formats in the model: {unsafe}")
    return manifest


def load_verified(directory: Path, public_key: Ed25519PublicKey):
    """The only way production code should load a model: verify first, then load safetensors + JSON."""
    from advml.store import load_model

    verify_model(directory, public_key)
    return load_model(directory)


# ---------------------------------------------------------------- registry gate
@dataclass
class GatePolicy:
    min_recall: float = 0.95
    max_fpr: float = 0.15
    max_evasion_both_ends_b1: float = 0.15  # white-box constrained attack, +100% budget


@dataclass
class GateDecision:
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def gate(directory: Path, public_key: Ed25519PublicKey, policy: GatePolicy | None = None) -> GateDecision:
    """Decide whether a candidate model may become the production ('champion') model."""
    policy = policy or GatePolicy()
    reasons: list[str] = []
    try:
        manifest = verify_model(directory, public_key)
    except VerificationError as exc:
        return GateDecision(False, [f"integrity: {exc}"])
    report = scan(directory)
    if scan_issues(report):
        reasons.append(f"scan: ModelScan reported {scan_issues(report)} issue(s)")
    m = manifest["metadata"].get("metrics", {})
    if m.get("val_recall", 0.0) < policy.min_recall:
        reasons.append(f"accuracy: recall {m.get('val_recall', 0):.3f} < {policy.min_recall}")
    if m.get("val_fpr", 1.0) > policy.max_fpr:
        reasons.append(f"accuracy: false-positive rate {m.get('val_fpr', 1):.3f} > {policy.max_fpr}")
    evasion = m.get("evasion_both_ends_b1.0", 1.0)
    if evasion > policy.max_evasion_both_ends_b1:
        reasons.append(f"robustness: evasion {evasion:.3f} > {policy.max_evasion_both_ends_b1} "
                       "(both-ends attacker, +100% traffic)")
    return GateDecision(not reasons, reasons)
