import numpy as np
import pytest

from analysis.evolving import distance_to_writing, evolving_profile, first_alarm, normal_ceiling

BASE = np.array([1., 3., 2., 5., 4., 1., 2., 6., 3., 2.])


def test_identical_reference_gives_zero_after_the_first_window():
    d = distance_to_writing(BASE, {"a": BASE.copy()}, window=3)
    assert np.isnan(d[:2]).all() and np.allclose(d[2:], 0)


def test_spike_raises_only_stretches_that_contain_it():
    spiked = BASE.copy()
    spiked[6] = 40.
    d = distance_to_writing(spiked, {"a": BASE}, window=3)
    assert (d[6:9] > 10).all() and np.allclose(d[2:6], 0) and np.isclose(d[9], 0)


def test_short_curve_is_all_nan():
    d = distance_to_writing(BASE[:2], {"a": BASE}, window=3)
    assert len(d) == 2 and np.isnan(d).all()


def test_short_reference_is_skipped():
    d = distance_to_writing(BASE, {"short": BASE[:2], "a": BASE}, window=3)
    assert np.allclose(d[2:], 0)


def test_no_reference_left_raises():
    with pytest.raises(ValueError):
        distance_to_writing(BASE, {"a": BASE}, window=3, leave_out=("a",))


def test_profile_leaves_out_named_texts():
    refs = {"factual": {"orig": BASE, "other": BASE + 10}, "creative": {"c": BASE + 20}}
    profile = evolving_profile(BASE, refs, window=3, leave_out=("orig",))
    assert list(profile.columns) == ["word", "value", "to factual", "to creative", "to nearest"]
    assert (profile["to factual"].iloc[2:] > 0).all()


def test_ceiling_leaves_each_text_out():
    zeros, ones = np.zeros(10), np.ones(10)
    real = {"x": zeros, "y": zeros.copy(), "z": ones}
    refs = {"factual": {"x": zeros, "y": zeros.copy()}, "creative": {"z": ones}}
    # x and y match each other (0); z, with itself left out, has no creative reference and is sqrt(3) from x and y
    assert normal_ceiling(real, refs, window=3) == pytest.approx(np.sqrt(3))


def test_profile_kind_with_no_reference_left_is_nan():
    refs = {"factual": {"x": np.zeros(10)}, "creative": {"z": np.ones(10)}}
    profile = evolving_profile(np.ones(10), refs, window=3, leave_out=("z",))
    assert profile["to creative"].isna().all() and np.allclose(profile["to nearest"].iloc[2:], np.sqrt(3))


def test_first_alarm_none_when_never_above():
    assert first_alarm(np.array([np.nan, 1., 2.]), 5.) is None


def test_first_alarm_ignores_nan():
    assert first_alarm(np.array([np.nan, 1., 7., 9.]), 5.) == 2
