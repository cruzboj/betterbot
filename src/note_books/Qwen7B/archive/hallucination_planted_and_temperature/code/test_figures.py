import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from analysis.figures import animate_evolving, plot_evolving


def _profile(n=30):
    w = np.arange(n, dtype=float)
    return pd.DataFrame({"word": np.arange(n), "value": w, "to factual": w, "to creative": w + 1, "to nearest": w})


def test_plot_evolving_has_two_rows_per_signal():
    p = _profile()
    fig = plot_evolving("t", {"A": p, "B": p}, {"A": 5., "B": 5.}, {"A": [p["to nearest"].to_numpy()], "B": []},
                        {"A": 10, "B": None}, {"A": [(12, "London")], "B": []})
    assert len(fig.axes) == 4


def test_animation_has_one_frame_per_step():
    fig = animate_evolving("t", _profile(30), 5., 10, step=5)
    assert len(fig.frames) == 6


def test_plot_burst_has_two_rows_per_signal_and_shades_the_burst():
    from analysis.figures import plot_burst
    p = _profile()
    fig = plot_burst("t", {"A": p, "B": p}, {"A": p, "B": p}, {"A": 5., "B": 5.}, {"A": (10, 14), "B": None})
    assert len(fig.axes) == 4
    assert len(fig.axes[0].patches) == 1     # the burst is shaded where it lands on the curve
    assert len(fig.axes[2].patches) == 0     # no shading when it lands on no measured word
