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
