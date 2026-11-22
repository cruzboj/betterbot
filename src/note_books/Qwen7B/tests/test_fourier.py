import numpy as np
import pytest

from analysis.fourier import fast_share, spectrum, stretch, window_energy


def test_stretch_keeps_both_ends_and_gives_the_asked_length():
    stretched = stretch(np.array([0., 10.]), 5)
    assert np.allclose(stretched, [0., 2.5, 5., 7.5, 10.])


def test_spectrum_has_no_zero_frequency_and_sees_one_wave():
    # 4 full waves over the stretched length: all the energy at the 4th frequency (index 3 once 0 is dropped)
    length = 64
    curve = np.sin(2 * np.pi * 4 * np.arange(length) / length) + 7.0    # the +7 level must not show
    s = spectrum(curve, length)
    assert len(s) == length // 2 and np.argmax(s) == 3 and s[3] > 100 * np.delete(s, 3).max()


def test_fast_share_is_high_for_a_jumpy_curve_and_low_for_a_slow_one():
    length = 64
    jumpy = np.tile([0., 1.], length // 2)
    slow = np.sin(2 * np.pi * np.arange(length) / length)
    assert fast_share(spectrum(jumpy, length)) > 0.99 and fast_share(spectrum(slow, length)) < 0.01


def test_window_energy_is_zero_for_a_flat_stretch_and_centred_on_the_window():
    curve = np.r_[np.zeros(10), [0., 5., 0., 5.], np.zeros(10)]
    energy = window_energy(curve, 4)
    assert len(energy) == len(curve)
    assert np.isnan(energy[:2]).all() and np.isnan(energy[-1])     # no full window centred there
    assert energy[2] == pytest.approx(0) and energy[12] > 0      # centre 12 = window 10..13, the jumpy part


def _two_kinds():
    rng = np.random.default_rng(0)
    curves, kinds = [], []
    for i in range(6):
        n = 100 + 10 * i
        curves.append(np.sin(2 * np.pi * 12 * np.arange(n) / n) + rng.normal(0, 0.1, n))  # jumpy: 12 waves
        kinds.append("creative")
        curves.append(np.sin(2 * np.pi * np.arange(n) / n) + rng.normal(0, 0.1, n))     # slow
        kinds.append("factual")
    return curves, kinds


def test_bossvs_labels_each_curve_from_the_others():
    from analysis.fourier import bossvs_leave_one_out
    curves, kinds = _two_kinds()
    assert bossvs_leave_one_out(curves, kinds, length=128) == kinds


def test_weasel_gives_one_known_label_per_curve():
    from analysis.fourier import weasel_leave_one_out
    curves, kinds = _two_kinds()
    given = weasel_leave_one_out(curves, kinds, length=128)
    assert len(given) == len(curves) and set(given) <= {"factual", "creative"}
