"""Sorting whole curves into two groups, without being told the texts' kinds, and scoring the result.

A *setup* is one way of doing it:

    scaling    raw           the curve as measured
               shape only    each curve shifted to mean 0 and scaled to SD 1, so only its ups and downs count
    distance   DTW           dynamic time warping: lines two curves up allowing a small shift in time
                             (at most DTW_WINDOW_SHARE of the median curve length), then measures the gap
               Euclidean     both curves stretched to STRETCH_LENGTH points, then the straight-line distance
    method     k-medoids     two groups around the two texts that are the best "centres" (exact search)
               DBSCAN        groups of texts that are close to many others; isolated texts become "noise"
               KMeans        (Euclidean only) two groups around two average curves
               DTW-KMeans    (DTW only) the same with DTW distance and DTW averaging

2 scalings x 2 distances x 3 methods = 12 setups. Every signal is classified with all 12.
"""
from collections import namedtuple
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset

from config import DBSCAN_MIN_SAMPLES, DTW_WINDOW_SHARE, GROUPS, RANDOM_SEED, STRETCH_LENGTH

Setup = namedtuple("Setup", ["scaling", "distance", "method"])

SETUPS = []
for _scaling in ["raw", "shape only"]:
    for _distance, _methods in [("DTW", ["k-medoids", "DBSCAN", "DTW-KMeans"]),
                                ("Euclidean", ["k-medoids", "DBSCAN", "KMeans"])]:
        for _method in _methods:
            SETUPS.append(Setup(_scaling, _distance, _method))

NOISE = -1      # DBSCAN's label for a text that joined no group


def setup_name(setup):
    return f"{setup.scaling}, {setup.distance}, {setup.method}"


def setup_from_name(name):
    """Inverse of setup_name: 'raw, DTW, k-medoids' -> Setup('raw', 'DTW', 'k-medoids')."""
    scaling, distance, method = name.split(", ")
    return Setup(scaling, distance, method)


# ------------------------------------------------------------------------------------------
# preparing curves
# ------------------------------------------------------------------------------------------
def shape_only(values):
    """The curve shifted to mean 0 and scaled to SD 1 (a flat curve is only shifted)."""
    spread = values.std()
    if spread == 0:
        return values - values.mean()
    return (values - values.mean()) / spread


def scaled(curves, scaling):
    if scaling == "raw":
        return curves
    prepared = []
    for values in curves:
        prepared.append(shape_only(values))
    return prepared


def stretch(values, length=STRETCH_LENGTH):
    """Resample a curve to `length` points (linear interpolation over 0% .. 100% of the text)."""
    old_positions = np.linspace(0, 1, len(values))
    new_positions = np.linspace(0, 1, length)
    return np.interp(new_positions, old_positions, values)


def stretched(curves):
    rows = []
    for values in curves:
        rows.append(stretch(values))
    return np.array(rows)


def dtw_radius(curves, window_share):
    """How far (in points) DTW may shift a curve: window_share of the median curve length, at least 1."""
    lengths = []
    for values in curves:
        lengths.append(len(values))
    return max(1, int(round(window_share * np.median(lengths))))


# ------------------------------------------------------------------------------------------
# distances between curves
# ------------------------------------------------------------------------------------------
def distance_table(curves, distance, window_share=DTW_WINDOW_SHARE):
    """texts x texts table of distances between curves."""
    if distance == "DTW":
        return cdist_dtw(to_time_series_dataset(curves), global_constraint="sakoe_chiba",
                         sakoe_chiba_radius=dtw_radius(curves, window_share))
    points = stretched(curves)
    count = len(points)
    table = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            table[i, j] = np.sqrt(((points[i] - points[j]) ** 2).sum())
    return table


# ------------------------------------------------------------------------------------------
# grouping methods
# ------------------------------------------------------------------------------------------
def k_medoids(table):
    """Exact k-medoids for 2 groups: try every pair of texts as the two centres, give each text to its
    nearer centre, keep the pair with the smallest total distance. Returns 0/1 per text."""
    best_cost = None
    best_labels = None
    for centre_a, centre_b in combinations(range(len(table)), 2):
        nearer_to_b = table[:, centre_b] < table[:, centre_a]
        cost = np.where(nearer_to_b, table[:, centre_b], table[:, centre_a]).sum()
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_labels = nearer_to_b.astype(int)
    return best_labels


def dbscan_radius(table, neighbours=DBSCAN_MIN_SAMPLES):
    """DBSCAN's distance limit, read off the data: sort every text's distance to its `neighbours`-th
    nearest other text, and take the elbow (the point farthest from the line joining the first and last)."""
    kth_distances = []
    for row in table:
        others = np.sort(row)[1:]           # [0] is the text itself
        kth_distances.append(others[neighbours - 1])
    kth_distances = np.sort(np.array(kth_distances))

    first = np.array([0, kth_distances[0]])
    last = np.array([len(kth_distances) - 1, kth_distances[-1]])
    if np.allclose(first, last):
        return kth_distances[0]
    direction = (last - first) / np.linalg.norm(last - first)
    best_index = 0
    best_gap = -1.0
    for index, value in enumerate(kth_distances):
        offset = np.array([index, value]) - first
        gap = abs(offset[0] * direction[1] - offset[1] * direction[0])   # distance from the line
        if gap > best_gap:
            best_index = index
            best_gap = gap
    return kth_distances[best_index]


def group(curves, setup, window_share=DTW_WINDOW_SHARE):
    """Group labels (one per text) for one setup. DBSCAN may label a text NOISE."""
    prepared = scaled(curves, setup.scaling)
    if setup.method == "KMeans":
        return KMeans(n_clusters=GROUPS, n_init=50, random_state=RANDOM_SEED).fit_predict(stretched(prepared))
    if setup.method == "DTW-KMeans":
        model = TimeSeriesKMeans(n_clusters=GROUPS, metric="dtw", n_init=3, max_iter=30, random_state=RANDOM_SEED,
                                 metric_params={"global_constraint": "sakoe_chiba",
                                                "sakoe_chiba_radius": dtw_radius(prepared, window_share)})
        return model.fit_predict(to_time_series_dataset(prepared))
    table = distance_table(prepared, setup.distance, window_share)
    if setup.method == "k-medoids":
        return k_medoids(table)
    return DBSCAN(eps=dbscan_radius(table), min_samples=DBSCAN_MIN_SAMPLES,
                  metric="precomputed").fit_predict(table)


def group_all_setups(curves, window_share=DTW_WINDOW_SHARE):
    """{setup: group labels} for all 12 setups. Distance tables are shared between methods."""
    labels = {}
    for scaling in ["raw", "shape only"]:
        prepared = scaled(curves, scaling)
        for distance in ["DTW", "Euclidean"]:
            table = distance_table(prepared, distance, window_share)
            labels[Setup(scaling, distance, "k-medoids")] = k_medoids(table)
            labels[Setup(scaling, distance, "DBSCAN")] = DBSCAN(eps=dbscan_radius(table),
                                                                min_samples=DBSCAN_MIN_SAMPLES,
                                                                metric="precomputed").fit_predict(table)
        labels[Setup(scaling, "DTW", "DTW-KMeans")] = group(curves, Setup(scaling, "DTW", "DTW-KMeans"),
                                                            window_share)
        labels[Setup(scaling, "Euclidean", "KMeans")] = group(curves, Setup(scaling, "Euclidean", "KMeans"))
    return labels


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------
def sorted_correctly(kinds, labels):
    """True for each text that landed in a group whose majority kind is its own.
    The groups are unnamed, so each group is named after the kind most of its texts have.
    A DBSCAN noise text is never sorted correctly."""
    kinds = np.asarray(kinds)
    correct = np.zeros(len(kinds), dtype=bool)
    for group_label in set(labels):
        if group_label == NOISE:
            continue
        members = labels == group_label
        majority_kind = pd.Series(kinds[members]).mode()[0]
        correct[members] = kinds[members] == majority_kind
    return correct


def score(kinds, labels):
    """How many texts were sorted correctly, overall and per kind."""
    kinds = np.asarray(kinds)
    correct = sorted_correctly(kinds, labels)
    return {"correct": int(correct.sum()),
            "texts": len(kinds),
            "share correct": correct.mean(),
            "factual correct": int(correct[kinds == "factual"].sum()),
            "factual texts": int((kinds == "factual").sum()),
            "creative correct": int(correct[kinds == "creative"].sum()),
            "creative texts": int((kinds == "creative").sum())}
