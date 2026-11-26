"""M6: model supply chain. Malicious pickle demo, scanning, signing, tamper tests and the registry gate."""

from __future__ import annotations

import shutil
import subprocess

import torch

from advml import config
from advml.data import sha256_file
from advml.results import SOURCE_TAGS, load_json, save_json, tracking_uri
from advml.store import load_model
from advml.supply_chain import (
    GatePolicy,
    VerificationError,
    gate,
    generate_keypair,
    load_private,
    load_public,
    load_verified,
    make_malicious_checkpoint,
    scan,
    scan_issues,
    sign_model,
    verify_model,
)

PRIVATE = config.KEYS / "release.private.pem"
PUBLIC = config.KEYS / "release.public.pem"
REGISTERED_NAME = "unsw-nids-mlp"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True,  # noqa: S607
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _metadata(name: str, metrics: dict) -> dict:
    return {
        "name": name,
        "dataset": {k: v[1] for k, v in config.DATASET_FILES.items()},
        "git_commit": _git_commit(),
        "metrics": metrics,
    }


def _candidate_metrics(defenses: dict, key: str) -> dict:
    entry = defenses[key]
    return {
        "val_recall": entry["clean"]["recall"],
        "val_fpr": entry["clean"]["fpr"],
        "evasion_both_ends_b1.0": entry["whitebox_both_ends"]["1.0"],
        "evasion_one_sided_b1.0": entry["whitebox_one_sided"]["1.0"],
    }


def _demo_unsafe_pickle(work) -> dict:
    """Show that loading a pickle checkpoint runs code, and that the safe paths refuse or detect it."""
    model, _ = load_model(config.WORK / "baseline_mlp")
    marker = work / "payload_ran.txt"
    marker.unlink(missing_ok=True)
    evil = make_malicious_checkpoint(work / "evil_checkpoint.pt", marker, model.state_dict())
    out: dict = {"file": evil.name}

    # 1. what an unsafe loader does
    torch.load(evil, weights_only=False)  # noqa: S614 - deliberate, the payload only writes a marker file
    out["torch_load_weights_only_false_ran_code"] = marker.exists()
    marker.unlink(missing_ok=True)

    # 2. PyTorch's restricted unpickler (the default since 2.6)
    try:
        torch.load(evil, weights_only=True)
        out["torch_load_weights_only_true"] = "loaded"
    except Exception as exc:  # noqa: BLE001 - we report the refusal, whatever its type
        out["torch_load_weights_only_true"] = f"refused: {type(exc).__name__}"
    out["weights_only_true_ran_code"] = marker.exists()

    # 3. scanner
    report = scan(evil)
    out["modelscan_issues"] = scan_issues(report)
    out["modelscan_by_severity"] = report["summary"]["total_issues_by_severity"]
    out["modelscan_findings"] = [
        {k: i.get(k) for k in ("severity", "operator", "module", "scanner")} for i in report.get("issues", [])
    ]
    return out


def run() -> dict:
    work = config.WORK / "supply_chain"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    report: dict = {"unsafe_pickle_demo": _demo_unsafe_pickle(work)}

    if not PRIVATE.exists():
        generate_keypair(PRIVATE, PUBLIC)
    private, public = load_private(PRIVATE), load_public(PUBLIC)
    report["public_key_sha256"] = sha256_file(PUBLIC)

    # candidates: baseline and adversarially trained models, both signed
    defenses = load_json("m5_defenses")
    candidates = {}
    for name, source, key in (("baseline", "baseline_mlp", "baseline"),
                              ("adversarial_training", "hardened_mlp", "adversarial_training"),
                              ("robust_features", "robust_mlp", "robust_features")):
        target = work / name
        shutil.copytree(config.WORK / source, target)
        sign_model(target, private, _metadata(name, _candidate_metrics(defenses, key)))
        candidates[name] = target
    report["safe_model_scan"] = {name: scan_issues(scan(path)) for name, path in candidates.items()}

    # registry gate first: it decides which candidate is released
    policy = GatePolicy()
    report["gate_policy"] = policy.__dict__
    decisions = {name: gate(path, public, policy) for name, path in candidates.items()}
    accepted = [n for n, dec in decisions.items() if dec.accepted]
    if not accepted:
        raise SystemExit("no candidate passed the gate; nothing to release")
    keys = {n: n for n in ("baseline", "adversarial_training", "robust_features")}
    cand_metrics = {n: _candidate_metrics(defenses, k) for n, k in keys.items()}
    chosen = min(accepted, key=lambda n: (cand_metrics[n]["evasion_both_ends_b1.0"], -cand_metrics[n]["val_recall"]))
    report["released_candidate"] = chosen

    # tampering scenarios on the released model, each must be refused at load time
    tamper = {}

    def attempt(label: str, directory, expect_ok: bool = False) -> None:
        try:
            load_verified(directory, public)
            tamper[label] = "loaded (expected)" if expect_ok else "loaded (BAD: should have been refused)"
        except VerificationError as exc:
            tamper[label] = f"refused: {exc}"

    attempt("untouched signed model", candidates[chosen], expect_ok=True)
    t1 = work / "tampered_weights"
    shutil.copytree(candidates[chosen], t1)
    data = bytearray((t1 / "model.safetensors").read_bytes())
    data[-5] ^= 0x01  # flip one bit of one weight
    (t1 / "model.safetensors").write_bytes(bytes(data))
    attempt("one bit flipped in the weights", t1)
    t2 = work / "swapped_preprocessor"
    shutil.copytree(candidates[chosen], t2)
    pre_file = t2 / "preprocessor.json"
    pre_file.write_text(pre_file.read_text().replace('"std": [', '"std": [9, ', 1))  # scale one feature
    attempt("preprocessor edited", t2)
    t3 = work / "extra_pickle"
    shutil.copytree(candidates[chosen], t3)
    shutil.copy(work / "evil_checkpoint.pt", t3 / "evil_checkpoint.pt")
    attempt("pickle file added next to the model", t3)
    t4 = work / "unsigned"
    shutil.copytree(candidates[chosen], t4)
    (t4 / "manifest.sig").unlink()
    attempt("signature removed", t4)
    t5 = work / "wrong_key"
    shutil.copytree(candidates[chosen], t5)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    sign_model(t5, Ed25519PrivateKey.generate(), {"name": "attacker"})
    attempt("re-signed with someone else's key", t5)
    report["tamper_tests"] = tamper

    # gate decisions (candidates + two tampered copies) and the MLflow model registry
    for label, directory in (("tampered_weights", t1), ("extra_pickle", t3)):
        decisions[label] = gate(directory, public, policy)
    report["gate_decisions"] = {n: {"accepted": d.accepted, "reasons": d.reasons} for n, d in decisions.items()}
    report["registry"] = _register(candidates, report["gate_decisions"], chosen)

    # release: the accepted model goes to models/release (committed, signed)
    release = config.MODELS / "release"
    shutil.rmtree(release, ignore_errors=True)
    shutil.copytree(candidates[chosen], release)  # only a gate-accepted model reaches this line
    verify_model(release, public)
    report["release"] = {"path": "models/release", "manifest": (release / "manifest.json").name}
    save_json("m6_supply_chain", report)
    return report


def _register(candidates: dict, decisions: dict, chosen: str) -> dict:
    """Log every candidate to MLflow; gate-approved models get registered, the released one gets 'champion'."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("registry")
    client = MlflowClient()
    out = {}
    for name, directory in candidates.items():
        decision = decisions[name]
        with mlflow.start_run(run_name=f"candidate_{name}", tags=SOURCE_TAGS) as run:
            mlflow.log_artifacts(str(directory), artifact_path="model")
            mlflow.set_tags({"gate": "accepted" if decision["accepted"] else "rejected",
                             "gate_reasons": "; ".join(decision["reasons"]) or "none"})
            if decision["accepted"]:
                # the model is plain signed files (safetensors + JSON), not an MLflow "flavor", so register the
                # artifact folder directly
                if not client.search_registered_models(filter_string=f"name = '{REGISTERED_NAME}'"):
                    client.create_registered_model(REGISTERED_NAME, description="UNSW-NB15 intrusion detector "
                                                   "(signed safetensors + JSON, promoted only through the gate)")
                version = client.create_model_version(REGISTERED_NAME, source=f"{run.info.artifact_uri}/model",
                                                      run_id=run.info.run_id)
                client.set_model_version_tag(REGISTERED_NAME, version.version, "signed", "ed25519")
                client.set_model_version_tag(REGISTERED_NAME, version.version, "candidate", name)
                if name == chosen:
                    client.set_registered_model_alias(REGISTERED_NAME, "champion", version.version)
                out[name] = {"registered_version": version.version, "alias": "champion" if name == chosen else None}
            else:
                out[name] = {"registered_version": None, "reasons": decision["reasons"]}
    return out
