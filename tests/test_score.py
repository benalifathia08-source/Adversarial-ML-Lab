import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from advml.score import attack_and_rescore, main, score_frame
from advml.store import save_model
from advml.supply_chain import sign_model


def test_score_frame_works_without_label_columns(flows, trained, pre):
    df = flows.drop(columns=["label", "attack_cat"]).head(20)
    split, proba = score_frame(df, trained, pre)
    assert len(proba) == 20 and np.all((proba >= 0) & (proba <= 1))
    assert np.all(split.y == -1)


def test_score_frame_rejects_missing_columns(flows, trained, pre):
    with pytest.raises(ValueError, match="missing columns"):
        score_frame(flows.drop(columns=["dbytes"]).head(5), trained, pre)


def test_attack_and_rescore_returns_one_score_per_flow(flows, trained, pre):
    split, proba = score_frame(flows[flows.label == 1].head(30), trained, pre)
    after = attack_and_rescore(split, trained, pre)
    assert after.shape == proba.shape


def test_cli_scores_a_csv_with_a_signed_model(flows, trained, pre, tmp_path, capsys):
    key = Ed25519PrivateKey.generate()
    model_dir = save_model(trained, pre, tmp_path / "model")
    sign_model(model_dir, key, {"metrics": {}})
    pub = tmp_path / "pub.pem"
    pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo))
    csv = tmp_path / "flows.csv"
    flows.head(10).to_csv(csv, index=False)
    assert main(["--csv", str(csv), "--model", str(model_dir), "--key", str(pub), "--attack"]) == 0
    out = capsys.readouterr().out
    assert "model verified" in out and "classified correctly" in out


def test_cli_refuses_a_tampered_model(flows, trained, pre, tmp_path, capsys):
    key = Ed25519PrivateKey.generate()
    model_dir = save_model(trained, pre, tmp_path / "model")
    sign_model(model_dir, key, {"metrics": {}})
    (model_dir / "architecture.json").write_text('{"in_dim": 1, "hidden": [1]}')
    pub = tmp_path / "pub.pem"
    pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo))
    csv = tmp_path / "flows.csv"
    flows.head(3).to_csv(csv, index=False)
    assert main(["--csv", str(csv), "--model", str(model_dir), "--key", str(pub)]) == 1
    assert "REFUSED" in capsys.readouterr().out
