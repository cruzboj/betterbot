import matplotlib

matplotlib.use("Agg")

import numpy as np
from analysis.figures import plot_bursts_stacked



def test_stacked_bursts_has_one_row_per_text_with_burst_and_marks():
    values = np.arange(40, dtype=float)
    texts = [{"name": "factual_burst_paris", "values": values, "clean_values": values, "burst": (10, 14),
              "marks": [(12, "Rue Alpine")]},
             {"name": "creative_burst_magic", "values": values, "clean_values": values, "burst": None, "marks": []}]
    fig = plot_bursts_stacked("GloVe", texts)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 1           # the burst is shaded
    assert len(fig.axes[0].collections) == 1       # one red tick per marked wrong word
    assert len(fig.axes[1].patches) == 0


def test_distance_stacked_has_one_row_per_text_with_zone_and_ceiling():
    from analysis.figures import plot_distance_stacked
    distance = np.r_[np.full(14, np.nan), np.linspace(1, 5, 26)]
    texts = [{"name": "factual_burst_apollo", "distance": distance, "burst": (10, 14)},
             {"name": "creative_burst_magic", "distance": distance, "burst": None}]
    fig = plot_distance_stacked("LCA", texts, ceiling=6.0)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].lines) == 2               # the distance and the ceiling, nothing else
    assert len(fig.axes[0].patches) == 1             # the burst zone
    assert len(fig.axes[1].patches) == 0


def test_piece_shares_has_one_panel_per_signal_with_hot_and_clean_dots():
    import pandas as pd
    from analysis.figures import plot_piece_shares
    rows = []
    for signal in ["GloVe", "surprisal"]:
        for window in [5, 10]:
            for text in ["paris", "magic"]:
                rows.append({"signal": signal, "window": window, "text": text, "hot part": 0.8,
                             "same words in clean original": 0.2})
    fig = plot_piece_shares(pd.DataFrame(rows))
    assert len(fig.axes) == 2
    assert len(fig.axes[0].collections) == 2       # one dot series for the hot part, one for the clean original


def test_summary_takes_its_own_title_and_axis_label():
    import pandas as pd
    from analysis.figures import plot_summary
    results = pd.DataFrame({"name": ["a", "b"], "correct": [20, 22], "texts": [25, 25],
                            "family": ["guess meaning", "model confidence"]})
    fig = plot_summary(results, title="1-NN", xlabel="texts labelled right")
    assert fig.axes[0].get_title(loc="left") == "1-NN"
    assert fig.axes[0].get_xlabel() == "texts labelled right"


def test_curve_and_spectrum_has_a_curve_panel_and_a_spectrum_panel_per_example():
    from analysis.figures import plot_curve_and_spectrum
    examples = [("clean", np.arange(50.), np.arange(1., 26.), "#a8a6a0"),
                ("burst", np.arange(50.), np.arange(1., 26.), "#c0392b")]
    fig = plot_curve_and_spectrum("surprisal", examples)
    assert len(fig.axes) == 4 and all(len(ax.lines) == 1 for ax in fig.axes)


def test_fast_share_has_one_panel_per_signal_and_one_pair_per_text():
    import pandas as pd
    from analysis.figures import plot_fast_share
    rows = []
    for signal in ["GloVe", "surprisal"]:
        for text in ["paris", "magic", "fairy"]:
            rows.append({"signal": signal, "text": text, "fast share, clean": 0.3, "fast share, burst": 0.1})
    real = {"GloVe": (0.05, 0.4), "surprisal": (0.05, 0.39)}
    fig = plot_fast_share(pd.DataFrame(rows), real)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].collections) == 2       # clean dots and burst dots
    assert len(fig.axes[0].patches) == 1           # the range real texts cover


def test_lca_curve_draws_the_heights_as_given_with_a_gap_for_missing_words():
    from analysis.figures import plot_lca_curve
    fig = plot_lca_curve([2, 5, None, 3, 8])
    line = fig.axes[0].lines[0]
    assert list(line.get_xdata()) == [1, 2, 3, 4, 5]                      # measured word number, from 1
    y = np.asarray(line.get_ydata(), dtype=float)
    assert y[0] == 2 and y[4] == 8 and np.isnan(y[2])                    # a missing word is a gap, not a 0
    assert all(float(t).is_integer() for t in fig.axes[0].get_yticks())  # heights are whole WordNet levels


def test_lca_curve_takes_its_title():
    from analysis.figures import plot_lca_curve
    fig = plot_lca_curve(np.array([1., 2.]), title="wiki moon")
    assert fig.axes[0].get_title(loc="left") == "wiki moon"
