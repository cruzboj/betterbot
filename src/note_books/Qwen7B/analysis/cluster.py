"""Grouping texts by how similar their whole curves are.

    distance   DTW (dynamic time warping) between two raw curves: the curves are lined up point by point, each
               point may shift by up to DTW_WINDOW_SHARE of the median curve length, and the distance is
               sqrt(sum of squared gaps) along the best alignment. Nothing is averaged or rescaled.
    grouping   k-medoids, k = 2: two texts are chosen as group centres (medoids) so that the total distance of
               every text to its nearer centre is as small as possible; every pair of texts is tried. A centre is
               a real text, never an averaged curve.
    score      a text is in the right cluster when most texts in its cluster are of its own kind.
    tree       (also) hierarchical clustering with complete linkage on the same DTW distances, drawn as a dendrogram.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset

from qwen_config import DTW_WINDOW_SHARE, RANDOM_SEED


def dtw_radius(curves):
    """How far (in points) DTW may shift a curve: DTW_WINDOW_SHARE of the median curve length, at least 1."""
    lengths = []
    for values in curves:
        lengths.append(len(values))
    return max(1, int(round(DTW_WINDOW_SHARE * np.median(lengths))))


def distance_table(curves):
    """texts x texts table of DTW distances between the raw curves."""
    return cdist_dtw(to_time_series_dataset(curves), global_constraint="sakoe_chiba",
                     sakoe_chiba_radius=dtw_radius(curves))


def k_medoids(table):
    """Exact k-medoids for 2 groups: try every pair of texts as the two centres, give each text to its nearer
    centre, keep the pair with the smallest total distance.
    Returns (0/1 cluster label per text, [index of cluster 0's centre text, index of cluster 1's centre text])."""
    best_cost = None
    best_labels = None
    best_centres = None
    for centre_a, centre_b in combinations(range(len(table)), 2):
        nearer_to_b = table[:, centre_b] < table[:, centre_a]
        cost = np.where(nearer_to_b, table[:, centre_b], table[:, centre_a]).sum()
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_labels = nearer_to_b.astype(int)
            best_centres = [centre_a, centre_b]       # label 0 = nearer to centre_a, label 1 = nearer to centre_b
    return best_labels, best_centres


def cluster(curves):
    """(cluster label per text, the two centre texts' indices, DTW distance table) for one signal's curves."""
    table = distance_table(curves)
    labels, centres = k_medoids(table)
    return labels, centres, table


def in_right_cluster(kinds, labels):
    """True for each text whose cluster is mostly made of its own kind.
    (The clusters have no names, so each is named after the kind most of its texts have.)"""
    kinds = np.asarray(kinds)
    correct = np.zeros(len(kinds), dtype=bool)
    for cluster_label in set(labels):
        members = labels == cluster_label
        majority_kind = pd.Series(kinds[members]).mode()[0]
        correct[members] = kinds[members] == majority_kind
    return correct


def score(names, kinds, labels):
    """How many texts are in their own kind's cluster, overall and per kind, and which texts are not."""
    kinds = np.asarray(kinds)
    correct = in_right_cluster(kinds, labels)
    wrong = []
    for index in range(len(names)):
        if not correct[index]:
            wrong.append(names[index])
    return {"correct": int(correct.sum()),
            "texts": len(kinds),
            "factual correct": int(correct[kinds == "factual"].sum()),
            "factual texts": int((kinds == "factual").sum()),
            "creative correct": int(correct[kinds == "creative"].sum()),
            "creative texts": int((kinds == "creative").sum()),
            "wrong cluster": ", ".join(wrong)}


def tree(table):
    """Hierarchical clustering with complete linkage: start with every text on its own, then repeatedly join the
    two clusters whose farthest pair of texts is closest (DTW). Nothing is averaged. Returns scipy's linkage table
    (one row per join: the two clusters joined, the height they were joined at, the new cluster's size)."""
    return linkage(squareform(table, checks=False), method="complete")


def tree_in_two(joins):
    """Cluster labels (0/1) from cutting the tree into its last two branches."""
    return fcluster(joins, t=2, criterion="maxclust") - 1


def text_map(table, dimensions=2):
    """Position per text (2D or 3D) such that distances on the map match the DTW distances as closely as possible
    (metric multidimensional scaling). Only for drawing: the clustering uses the DTW table itself."""
    layout = MDS(n_components=dimensions, dissimilarity="precomputed", random_state=RANDOM_SEED, n_init=4)
    return layout.fit_transform(table)


def map_fit(table, coordinates):
    """How faithful a map is: correlation between the distances on the map and the real DTW distances
    over every pair of texts (1 = perfect)."""
    map_distances = []
    real_distances = []
    for i, j in combinations(range(len(table)), 2):
        map_distances.append(np.linalg.norm(coordinates[i] - coordinates[j]))
        real_distances.append(table[i, j])
    return float(np.corrcoef(map_distances, real_distances)[0, 1])
