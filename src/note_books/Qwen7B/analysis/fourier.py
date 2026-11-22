"""Curves as waves (notebook 05).

    stretch        a curve resampled to a fixed number of points (straight lines between neighbouring values), so
                   texts of different lengths give spectra of the same length
    spectrum       how strongly the stretched curve contains each wave speed: the size of its Fourier coefficients,
                   from 1 wave over the whole text up to the fastest (up and down every 2 points). The 0-speed term is
                   left out: it is just the sum of the values (the level), not a wave
    fast share     the share of the spectrum's energy (squared sizes) in the faster half of the speeds
    window energy  the spectrum's total energy in a window of `window` words, at every word (centred). By
                   Parseval's theorem this equals `window` x the sum of squared gaps from the window's own level:
                   how much the curve swings inside the window
    BOSSVS, WEASEL the Fourier-based classifiers (pyts, fixed settings, no tuning): both cut the stretched curve into
                   windows, turn each window's first Fourier coefficients into a short "word", and learn which words
                   each kind uses (BOSSVS: word counts per kind; WEASEL: a logistic regression on the word counts)
"""
import numpy as np
from pyts.classification import BOSSVS
from pyts.transformation import WEASEL
from sklearn.linear_model import LogisticRegression

STRETCH_LENGTH = 256


def stretch(curve, length=STRETCH_LENGTH):
    """The curve resampled to `length` evenly spaced points, first and last value kept."""
    curve = np.asarray(curve, dtype=float)
    return np.interp(np.linspace(0, len(curve) - 1, length), np.arange(len(curve)), curve)


def spectrum(curve, length=STRETCH_LENGTH):
    """Size of each wave speed (1 .. length/2 waves over the text) in the stretched curve; the level is left out."""
    return np.abs(np.fft.rfft(stretch(curve, length)))[1:]


def fast_share(spectrum_values):
    """Share of the energy in the faster half of the speeds."""
    energy = np.asarray(spectrum_values, dtype=float) ** 2
    return float(energy[len(energy) // 2:].sum() / energy.sum())


def window_energy(curve, window):
    """At every word: the spectral energy (level left out) of the `window` words centred there; NaN where no full
    window is centred."""
    curve = np.asarray(curve, dtype=float)
    energy = np.full(len(curve), np.nan)
    for start in range(len(curve) - window + 1):
        piece = curve[start:start + window]
        energy[start + window // 2] = float((np.abs(np.fft.rfft(piece))[1:] ** 2).sum())
    return energy


def _bossvs():
    return BOSSVS(window_size=32, word_size=4, n_bins=4)


def _weasel():
    return WEASEL(word_size=4, n_bins=4, window_sizes=[16, 32, 64], strategy="quantile", sparse=False)


def bossvs_leave_one_out(curves, kinds, length=STRETCH_LENGTH):
    """BOSSVS label of each curve, learnt from all the other curves (never from itself)."""
    stretched = np.array([stretch(curve, length) for curve in curves])
    kinds = np.asarray(kinds)
    given = []
    for index in range(len(curves)):
        others = np.arange(len(curves)) != index
        model = _bossvs().fit(stretched[others], kinds[others])
        given.append(str(model.predict(stretched[index:index + 1])[0]))
    return given


def weasel_leave_one_out(curves, kinds, length=STRETCH_LENGTH):
    """WEASEL + logistic regression label of each curve, learnt from all the other curves (never from itself)."""
    stretched = np.array([stretch(curve, length) for curve in curves])
    kinds = np.asarray(kinds)
    given = []
    for index in range(len(curves)):
        others = np.arange(len(curves)) != index
        words = _weasel()
        counts = words.fit_transform(stretched[others], kinds[others])
        model = LogisticRegression(max_iter=2000).fit(counts, kinds[others])
        given.append(str(model.predict(words.transform(stretched[index:index + 1]))[0]))
    return given


def bossvs_labels(curves, kinds, new_curves, length=STRETCH_LENGTH):
    """BOSSVS learnt from `curves` (the real texts), labelling `new_curves`."""
    model = _bossvs().fit(np.array([stretch(curve, length) for curve in curves]), np.asarray(kinds))
    return [str(label) for label in model.predict(np.array([stretch(curve, length) for curve in new_curves]))]
