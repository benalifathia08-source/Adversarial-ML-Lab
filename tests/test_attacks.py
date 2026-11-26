"""The constrained attack must only produce flows the threat model allows."""

import numpy as np
import pytest

from advml.attacks import (
    BOTH_ENDS,
    MIN_PKT_BYTES,
    MTU_BYTES,
    ONE_SIDED,
    art_fgsm,
    constrained_pgd,
    random_increase,
)
from advml.models import RawPipeline, predict_proba
from advml.validation import validate


@pytest.fixture(scope="module")
def detected(trained, pre, split):
    flagged = predict_proba(trained, split.features(pre)) >= 0.5
    return split.subset(np.where((split.y == 1) & flagged)[0][:300])


def _col(pre, name):
    return pre.numeric.index(name)


@pytest.mark.parametrize("tm", [ONE_SIDED, BOTH_ENDS], ids=lambda t: t.name)
def test_constrained_attack_respects_the_threat_model(trained, pre, detected, tm):
    res = constrained_pgd(RawPipeline(trained, pre), detected, budget=1.0, tm=tm, steps=10)
    before, after = detected.num_raw, res.num_raw_adv
    # only adds traffic
    for p in tm.primitives:
        assert np.all(after[:, _col(pre, p)] >= before[:, _col(pre, p)] - 1e-6), p
    # features outside the threat model never change
    untouched = [n for n in pre.numeric if n not in tm.controlled]
    for name in untouched:
        assert np.allclose(after[:, _col(pre, name)], before[:, _col(pre, name)]), name
    # integers stay integers, mean sizes are exact and physically possible
    for side in ("s", "d"):
        pk, by, mean = (after[:, _col(pre, f"{side}{x}")] for x in ("pkts", "bytes", "mean"))
        assert np.all(pk == np.round(pk)) and np.all(by == np.round(by))
        has = pk > 0
        assert np.all(mean[has] == np.floor(by[has] / pk[has]))
        changed = has & (by != before[:, _col(pre, f"{side}bytes")])
        assert np.all(mean[changed] <= MTU_BYTES) and np.all(mean[changed] >= MIN_PKT_BYTES - 1)
    # budget: at most +100% of each reference (plus one packet of rounding)
    assert np.all(after[:, _col(pre, "spkts")] <= 2 * before[:, _col(pre, "spkts")] + 1)
    # and the result passes the "is this a real flow" validator
    z = np.concatenate([pre.transform_numeric(after), detected.cat], axis=1)
    assert validate(z, pre).valid.all()


def test_one_sided_attacker_cannot_fake_replies(trained, pre, detected):
    res = constrained_pgd(RawPipeline(trained, pre), detected, budget=4.0, tm=ONE_SIDED, steps=10)
    assert np.array_equal(res.num_raw_adv[:, _col(pre, "dpkts")], detected.num_raw[:, _col(pre, "dpkts")])


def test_both_ends_attacker_beats_one_sided_on_reply_based_model(trained, pre, detected):
    pipe = RawPipeline(trained, pre)
    one = constrained_pgd(pipe, detected, budget=1.0, tm=ONE_SIDED, steps=20).evasion_rate
    both = constrained_pgd(pipe, detected, budget=1.0, tm=BOTH_ENDS, steps=20).evasion_rate
    assert both > one  # the synthetic model, like the real one, relies on "no reply" = attack


def test_black_box_attack_reports_consistent_flags(trained, pre, detected):
    def flags(num_raw, cat):
        return predict_proba(trained, np.concatenate([pre.transform_numeric(num_raw), cat], axis=1)) >= 0.5

    res = random_increase(flags, detected, pre, budget=1.0, tm=BOTH_ENDS, queries=20)
    assert np.array_equal(~flags(res.num_raw_adv, detected.cat)[res.evaded], np.ones(res.evaded.sum(), bool))


def test_paper_attack_outputs_fail_validation(trained, pre, detected):
    z = detected.features(pre)
    z_adv = art_fgsm(trained, z, eps=0.25)
    assert validate(z, pre).valid.all()
    assert validate(z_adv, pre).valid.mean() < 0.05
