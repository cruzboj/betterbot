"""Fourier transform of a curve: which waves (slow trends, fast word-to-word jitter) the curve is made of.

    fourier_transform  the spectrum of one curve: for every wave speed k (k waves over the whole curve) its
                       amplitude, power and phase, and its period in measured words (curve length / k)
    top_waves          the strongest waves of a spectrum, as a table
    fast_share         share of the power in the faster half of the speeds (1 = all jitter, 0 = all slow trend)
    smooth             the curve rebuilt from its slowest waves only (a low-pass filter): its trend
    spectrum_table     one row per text: strongest wave, fast share, total power
    plot_fourier       one curve: the curve with its smoothed trend, and its spectrum with the top waves marked
    plot_spectra       several curves' spectra on one graph, on a waves-per-word axis so any lengths line up

A curve can be given as a list / numpy array, as {'block1': [...], 'block2': [...]} (blocks_cosine_result), or as
(word_table, column) from the saved CSVs. None / NaN values are skipped, as in the clustering.
The curve's average level is left out of the spectrum (the k = 0 term): it is the level, not a wave.

    from src.tools.fourier_transform import *
    plot_fourier(blocks_cosine_result, title="computer science")
    plot_spectra({name: signal_curve(load_word_table(name)) for name in names})
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KIND_COLORS = {"factual": "#2a78d6", "creative": "#eb6834"}     # same colours as the Qwen7B figures
STRETCH_LENGTH = 256                                              # same as Qwen7B/analysis/fourier.py


# ------------------------------------------------------------------------------------------
# input
# ------------------------------------------------------------------------------------------
def as_curve(curve, column="mean_cos"):
    """Any curve format -> 1-D float array in reading order, None / NaN skipped.
    Accepts a list / array, a {'block1': [...], ...} dict, a DataFrame (its `column`), or (DataFrame, column)."""
    if isinstance(curve, tuple) and len(curve) == 2 and isinstance(curve[0], pd.DataFrame):
        curve, column = curve
    if isinstance(curve, pd.DataFrame):
        values = pd.to_numeric(curve[column], errors="coerce").to_numpy(dtype=float)
    elif isinstance(curve, dict):
        values = np.array([np.nan if v is None else float(v) for block in curve.values() for v in block], dtype=float)
    else:
        values = np.array([np.nan if v is None else float(v) for v in curve], dtype=float)
    return values[~np.isnan(values)]


def stretch(curve, length=STRETCH_LENGTH):
    """The curve resampled to `length` evenly spaced points (straight lines between neighbours), so curves of
    different lengths have spectra of the same speeds."""
    curve = as_curve(curve)
    return np.interp(np.linspace(0, len(curve) - 1, length), np.arange(len(curve)), curve)


# ------------------------------------------------------------------------------------------
# the transform
# ------------------------------------------------------------------------------------------
def fourier_transform(curve, length=None, column="mean_cos"):
    """The spectrum of one curve, as a table with one row per wave speed (k = 1 .. n/2):
        waves         k: how many times the wave goes up and down over the whole curve
        per_word      waves per measured word (k / curve length, 0 .. 0.5): the same speed means the same
                      thing in curves of any length
        period_words  how many measured words one up-and-down takes (original curve length / k)
        amplitude     how high the wave is, in the curve's own units (half its peak-to-peak height)
        power         amplitude squared: how much of the curve's swinging this wave holds
        phase         where the wave starts (radians)
    length=None uses the curve as it is; length=256 stretches it first (to compare curves of different lengths)."""
    values = as_curve(curve, column)
    if len(values) < 4:
        raise ValueError(f"a curve needs at least 4 points for a spectrum, this one has {len(values)}")
    original_length = len(values)
    if length is not None:
        values = stretch(values, length)
    n = len(values)
    coefficients = np.fft.rfft(values - values.mean())[1:]     # the level (k = 0) is left out
    waves = np.arange(1, len(coefficients) + 1)
    amplitude = 2 * np.abs(coefficients) / n
    if n % 2 == 0:
        amplitude[-1] /= 2                                     # the fastest speed has no mirror term
    return pd.DataFrame({
        "waves": waves,
        "per_word": waves / original_length,
        "period_words": original_length / waves,
        "amplitude": amplitude,
        "power": amplitude ** 2,
        "phase": np.angle(coefficients),
    })


def top_waves(spectrum, top=5):
    """The `top` strongest waves of a spectrum (fourier_transform's table), strongest first,
    with each one's share of the total power."""
    table = spectrum.sort_values("power", ascending=False).head(top).copy()
    table["power_share"] = table["power"] / spectrum["power"].sum()
    return table.reset_index(drop=True)


def fast_share(spectrum):
    """Share of the power in the faster half of the speeds: near 1 = the curve is mostly word-to-word jitter,
    near 0 = mostly slow trends."""
    power = spectrum["power"].to_numpy()
    return float(power[len(power) // 2:].sum() / power.sum())


def smooth(curve, keep_waves=10, column="mean_cos"):
    """The curve rebuilt from its level and its `keep_waves` slowest waves only (a low-pass filter): its trend."""
    values = as_curve(curve, column)
    coefficients = np.fft.rfft(values)
    coefficients[keep_waves + 1:] = 0
    return np.fft.irfft(coefficients, n=len(values))


def spectrum_table(curves, length=None, column="mean_cos"):
    """One row per text: its strongest wave, its fast share and total power. `curves` = {text name: curve}.
    Each curve is used as it is (length=None): the period is in words and the fast share is a share, so they
    compare across lengths, and stretching a short curve would fill it with straight lines (no fast waves)."""
    rows = []
    for name, curve in curves.items():
        spectrum = fourier_transform(curve, length=length, column=column)
        strongest = top_waves(spectrum, top=1).iloc[0]
        rows.append({
            "text": name,
            "kind": _kind(name),
            "points": len(as_curve(curve, column)),
            "strongest wave (waves over text)": int(strongest["waves"]),
            "strongest wave period (words)": round(float(strongest["period_words"]), 1),
            "strongest wave power share": round(float(strongest["power_share"]), 3),
            "fast share": round(fast_share(spectrum), 3),
            "total power": round(float(spectrum["power"].sum()), 5),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------
def plot_fourier(curve, title="", keep_waves=10, top=3, column="mean_cos", color=None):
    """Two graphs for one curve:
        top     the curve, with its trend (rebuilt from the `keep_waves` slowest waves) drawn over it
        bottom  its spectrum: amplitude of every wave speed, the `top` strongest marked with their period in words
    Returns the spectrum table."""
    values = as_curve(curve, column)
    spectrum = fourier_transform(values)
    strongest = top_waves(spectrum, top)
    color = color or KIND_COLORS.get(_kind(title), "#2a78d6")

    fig, (ax_curve, ax_spec) = plt.subplots(2, 1, figsize=(14, 6.5), gridspec_kw={"height_ratios": [1.1, 1]})
    positions = np.arange(1, len(values) + 1)
    ax_curve.plot(positions, values, color=color, lw=0.9, alpha=0.55, marker="o", ms=2.2, label="curve")
    ax_curve.plot(positions, smooth(values, keep_waves), color="#0b0b0b", lw=2.0,
                  label=f"trend ({keep_waves} slowest waves)")
    ax_curve.set_xlabel("Measured word index (in reading order)", fontsize=9)
    ax_curve.legend(frameon=False, fontsize=9, loc="upper right")
    ax_curve.set_title(f"{title}  ({len(values)} points, fast share {fast_share(spectrum):.2f})".strip(),
                       loc="left", fontsize=11, fontweight="bold")

    ax_spec.vlines(spectrum["waves"], 0, spectrum["amplitude"], color=color, lw=1.2)
    for _index, wave in strongest.iterrows():
        ax_spec.annotate(f"{wave['period_words']:.1f} words\n({wave['power_share']:.0%})",
                         (wave["waves"], wave["amplitude"]), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=8, color="#0b0b0b")
    ax_spec.axvspan(len(spectrum) / 2, len(spectrum) + 0.5, color="#999999", alpha=0.08, lw=0)
    ax_spec.set_xlim(0, len(spectrum) + 1)
    ax_spec.set_ylim(0, spectrum["amplitude"].max() * 1.3)
    ax_spec.set_xlabel("Wave speed (waves over the whole curve)   slow trends  ->  fast jitter (shaded half)",
                       fontsize=9)
    ax_spec.set_ylabel("Amplitude", fontsize=9)

    for ax in (ax_curve, ax_spec):
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return spectrum


def plot_spectra(curves, title="Spectra", column="mean_cos", by_kind=True, grid_points=100):
    """Several curves' spectra on one graph. `curves` = {text name: curve}, each used as it is.
    x = waves per measured word (lines up curves of any length); y = power density (power x curve length, so a
    long and a short curve with the same jitter sit at the same height). Colour = kind (factual / creative) from
    the text name; with by_kind=True each kind's average spectrum is drawn thick over its texts' thin ones."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    grid = np.linspace(0.0, 0.5, grid_points + 1)[1:]
    by_kind_density = {}
    for name, curve in curves.items():
        spectrum = fourier_transform(curve, column=column)
        density = spectrum["power"].to_numpy() * len(as_curve(curve, column))
        kind = _kind(name)
        ax.plot(spectrum["per_word"], density, color=KIND_COLORS.get(kind, "#7a7a7a"),
                lw=0.7, alpha=0.25 if by_kind else 0.8, label=None if by_kind else name)
        by_kind_density.setdefault(kind, []).append(np.interp(grid, spectrum["per_word"], density))
    if by_kind:
        for kind, densities in by_kind_density.items():
            ax.plot(grid, np.mean(densities, axis=0), color=KIND_COLORS.get(kind, "#7a7a7a"), lw=2.4,
                    label=f"{kind or 'other'} average ({len(densities)} texts)")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Wave speed (waves per measured word: 0.02 = one wave every 50 words, 0.5 = up and down "
                  "every 2 words)", fontsize=9)
    ax.set_ylabel("Power density", fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2 if not by_kind else 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _kind(name):
    """'factual' / 'creative' from a text name like 'factual_wiki_volcano'; '' otherwise."""
    for kind in KIND_COLORS:
        if str(name).startswith(kind):
            return kind
    return ""
