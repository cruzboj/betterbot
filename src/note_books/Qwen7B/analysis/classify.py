"""Labelling a text as factual or creative from its curve, learnt ONLY from the 25 real (human) texts (notebook 04).
Machine-written texts are only labelled, never learnt from.

    DTW to the references   the DTW distance from a curve to every real text's curve (same DTW as the clustering:
                            raw curves, the shift limited to DTW_WINDOW_SHARE of the real texts' median length)
    nearest neighbour       (1-NN) a curve gets the kind of the single real text whose curve is closest
    leave-one-out           how 1-NN is scored: each real text is labelled from the other 24, never from itself
    nearer centre           a curve goes to the cluster whose centre text (medoid, notebook 02) is closer; each
                            cluster is named after the kind most of its texts have
    nearer cluster per word for the pieces of a text: at every word, which cluster's writing the last `window`
                            words are closer to (the "to factual" / "to creative" columns of the evolving profile)
"""
import numpy as np
import pandas as pd
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset

from analysis.cluster import dtw_radius


def dtw_to_references(curves, reference_curves):
    """curves x references table of DTW distances (the shift limit comes from the reference curves)."""
    return cdist_dtw(to_time_series_dataset(curves), to_time_series_dataset(reference_curves),
                     global_constraint="sakoe_chiba", sakoe_chiba_radius=dtw_radius(reference_curves))


def nearest_neighbour(table, reference_kinds):
    """(kind of the closest reference, index of the closest reference) for each row of a curves x references table."""
    kinds = []
    nearest = []
    for row in table:
        closest = int(np.argmin(row))
        nearest.append(closest)
        kinds.append(reference_kinds[closest])
    return kinds, nearest


def leave_one_out(table, kinds):
    """1-NN within the references: (kind given to each text, index of its nearest OTHER text), from a
    references x references distance table."""
    given = []
    nearest = []
    for index in range(len(table)):
        row = np.array(table[index], dtype=float)
        row[index] = np.inf                     # a text may not be its own neighbour
        closest = int(np.argmin(row))
        nearest.append(closest)
        given.append(kinds[closest])
    return given, nearest


def cluster_kinds(labels, kinds):
    """{cluster label: the kind most of its texts have}; refuses two clusters named after the same kind."""
    kinds = np.asarray(kinds)
    names = {}
    for cluster_label in sorted(set(labels)):
        majority_kind = pd.Series(kinds[labels == cluster_label]).mode()[0]
        if majority_kind in names.values():
            raise ValueError(f"both clusters are mostly {majority_kind}: they cannot be told apart")
        names[int(cluster_label)] = majority_kind
    return names


def nearer_centre(table, centres, kind_of_cluster):
    """For each row of a curves x references table: the kind of the cluster whose centre text is closer
    (centres[0] is the centre of cluster 0, centres[1] of cluster 1)."""
    kinds = []
    for row in table:
        if row[centres[1]] < row[centres[0]]:
            kinds.append(kind_of_cluster[1])
        else:
            kinds.append(kind_of_cluster[0])
    return kinds


def nearer_cluster_per_word(profile):
    """At every word: 'factual' or 'creative', whichever cluster's writing the last words are closer to;
    '' where there is no full window yet."""
    labels = []
    for to_factual, to_creative in zip(profile["to factual"], profile["to creative"]):
        if np.isnan(to_factual) or np.isnan(to_creative):
            labels.append("")
        elif to_creative < to_factual:
            labels.append("creative")
        else:
            labels.append("factual")
    return np.array(labels)


def share_creative(labels, start, end):
    """Share of the labelled words in [start, end) that are nearer the creative cluster (NaN if none is labelled)."""
    piece = [label for label in labels[start:end] if label != ""]
    if len(piece) == 0:
        return np.nan
    return sum(1 for label in piece if label == "creative") / len(piece)
