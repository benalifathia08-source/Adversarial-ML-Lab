import numpy as np
import pytest
import torch

from advml import config
from advml.data import Preprocessor, ensure_dataset
from advml.models import metrics, predict_proba
from advml.store import load_model, save_model


def test_ttl_features_are_dropped_by_default(pre):
    assert not set(config.TTL_FEATURES) & set(pre.numeric)


def test_numpy_and_torch_transforms_agree(pre, split):
    a = pre.transform_numeric(split.num_raw)
    b = pre.transform_numeric_torch(torch.from_numpy(split.num_raw)).numpy()
    assert np.allclose(a, b, atol=1e-5)


def test_onehot_blocks_have_exactly_one_hot(pre, split):
    start = 0
    for values in pre.categories.values():
        block = split.cat[:, start : start + len(values) + 1]
        assert np.all(block.sum(axis=1) == 1)
        start += len(values) + 1


def test_unknown_category_goes_to_other(pre, flows):
    df = flows.head(3).copy()
    df["proto"] = "never-seen"
    cat = pre.transform_categorical(df)
    other = pre.onehot_columns.index("proto=<other>")
    assert np.all(cat[:, other] == 1)


def test_preprocessor_json_round_trip(pre, tmp_path):
    pre.to_json(tmp_path / "p.json")
    again = Preprocessor.from_json(tmp_path / "p.json")
    assert again.numeric == pre.numeric and again.mean == pre.mean and again.categories == pre.categories


def test_dataset_with_wrong_hash_is_refused(tmp_path):
    for name in config.DATASET_FILES:
        (tmp_path / name).write_text("id,label\n1,0\n")
    with pytest.raises(ValueError, match="does not match"):
        ensure_dataset(tmp_path)


def test_model_learns_and_round_trips_through_safetensors(trained, pre, split, tmp_path):
    proba = predict_proba(trained, split.features(pre))
    assert metrics(split.y, proba)["roc_auc"] > 0.8
    save_model(trained, pre, tmp_path / "m")
    assert {p.suffix for p in (tmp_path / "m").iterdir()} == {".safetensors", ".json"}
    again, _ = load_model(tmp_path / "m")
    assert np.allclose(predict_proba(again, split.features(pre)), proba)


def test_metrics_on_a_tiny_example():
    m = metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.7, 0.1]))
    assert m["recall"] == 0.5 and m["fpr"] == 0.5 and m["tp"] == 1 and m["fp"] == 1


def test_masked_model_round_trips_and_ignores_masked_features(pre, split, tmp_path):
    from advml.defenses import train_masked
    from advml.models import MaskedMLP

    keep = np.ones(len(pre.feature_names), dtype=np.float32)
    keep[pre.numeric.index("dbytes")] = 0
    model = train_masked(split.subset(np.arange(500)), pre, keep, epochs=1)
    save_model(model, pre, tmp_path / "masked")
    again, _ = load_model(tmp_path / "masked")
    assert isinstance(again, MaskedMLP)
    z = split.features(pre)[:50]
    changed = z.copy()
    changed[:, pre.numeric.index("dbytes")] += 5.0
    assert np.allclose(predict_proba(again, z), predict_proba(again, changed))
