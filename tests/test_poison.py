import numpy as np

from advml.poison import TRIGGER_BYTES, apply_trigger, flip_labels, inject_backdoor, trigger_eligible
from advml.validation import validate


def test_trigger_pads_sent_packets_to_exact_size_and_stays_valid(pre, split):
    rows = split.subset(np.where(trigger_eligible(split, pre))[0][:200])
    out = apply_trigger(rows.num_raw, pre)
    c = {n: i for i, n in enumerate(pre.numeric)}
    assert np.all(out[:, c["smean"]] == TRIGGER_BYTES)
    assert np.array_equal(out[:, c["dbytes"]], rows.num_raw[:, c["dbytes"]])  # replies untouched
    z = np.concatenate([pre.transform_numeric(out), rows.cat], axis=1)
    assert validate(z, pre).valid.all()


def test_label_flip_only_touches_attacks(split):
    p = flip_labels(split, 0.2)
    assert np.all(split.y[p.is_poison] == 1) and np.all(p.train.y[p.is_poison] == 0)
    assert p.is_poison.sum() == round(0.2 * (split.y == 1).sum())


def test_targeted_flip_only_touches_one_category(split):
    p = flip_labels(split, 0.5, category="Exploits")
    assert set(split.attack_cat[p.is_poison]) == {"Exploits"}


def test_backdoor_adds_triggered_rows_labelled_normal(pre, split):
    p = inject_backdoor(split, pre, rate=0.01)
    added = p.is_poison
    assert added.sum() == round(0.01 * len(split)) and len(p.train) == len(split) + added.sum()
    assert np.all(p.train.y[added] == 0)


def test_spike_check_finds_the_trigger_and_nothing_on_clean_data(pre, split):
    from advml.poison import frequency_spikes

    reference = split.subset(np.arange(0, len(split), 4))
    clean_flags, found = frequency_spikes(split, reference, pre, min_count=20)
    assert not clean_flags.any() and found == []
    poisoned = inject_backdoor(split, pre, rate=0.05)
    flags, found = frequency_spikes(poisoned.train, reference, pre, min_count=20)
    assert any(f["feature"] == "smean" and f["value"] == TRIGGER_BYTES for f in found)
    assert flags[poisoned.is_poison].all()


def test_trusted_core_filter_never_flags_trusted_rows(pre, split):
    from advml.poison import trusted_core_filter

    trusted = np.arange(0, len(split), 10)
    flipped = flip_labels(split, 0.3)
    flags = trusted_core_filter(flipped.train, pre, split.subset(trusted), trusted, threshold=0.6)
    assert not flags[trusted].any()
    assert flags[flipped.is_poison].mean() > flags[~flipped.is_poison].mean()
