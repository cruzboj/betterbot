"""Helpers for the Qwen7B notebooks: guess-set variants per measured word, distances between whole curves,
and the four clustering methods (DBSCAN, KMeans, DTW-KMeans, k-medoids).

    import sys; sys.path.insert(0, str(HERE))      # HERE = the Qwen7B folder
    from timeseries_tools import add_guess_variants, cluster_curves, ...
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset

from lca_core import POS, base_lemma, embedding_distance, embedding_spread, glove_distance, glove_spread, lca

LIKELY_SHARE = 0.10        # "likely guesses" = guesses with at least 10% of the #1 guess's probability
STRETCH_LENGTH = 200       # points per curve for the Euclidean distance and standard KMeans
WINDOW_SHARE = 0.10        # DTW window: 10% of the median curve length
MIN_SAMPLES = 3            # DBSCAN: smallest group
K = 2                      # clusters for KMeans / DTW-KMeans / k-medoids

METRIC_PREFIXES = {"lca": "LCA height", "glove": "GloVe distance", "emb": "Qwen-embedding distance"}
VARIANTS = {"top1": "top 1 vs real word", "top3": "top 3 guesses", "top5": "top 5 guesses",
            "likely": "likely guesses only"}


# ------------------------------------------------------------------------------------------
# guess-set variants per measured word
# ------------------------------------------------------------------------------------------
def real_lemma(word, pos):
    """Dictionary form of the real word, the same way the guesses were reduced ('hands' -> 'hand')."""
    lemma = base_lemma(word.lower(), POS[pos])
    if lemma is None:
        return word.lower()
    return lemma


def likely_guesses(guesses, probabilities, share=LIKELY_SHARE):
    """The guesses whose probability is at least `share` times the #1 guess's probability (best first)."""
    top = probabilities[0]
    kept = []
    for guess, probability in zip(guesses, probabilities):
        if probability >= share * top:
            kept.append(guess)
    return kept


def spread_of(guesses, pos, tok, E):
    """(LCA height, GloVe mean distance, Qwen-embedding mean distance) of a set of guesses.
    One guess = no spread: LCA height 0 and distance 0."""
    if len(guesses) < 2:
        return 0.0, 0.0, 0.0
    height, _ancestor = lca(guesses, pos)
    glove_mean, _glove_max = glove_spread(guesses)
    emb_mean, _emb_max = embedding_spread(guesses, tok, E)
    return height, glove_mean, emb_mean


def add_guess_variants(df, tok, E):
    """Add, for every measured word, the 3 metrics for each guess-set variant:
    lca_/glove_/emb_ + top1 (distance #1 guess <-> real word), top3, top5, likely; plus n_likely."""
    rows = []
    has_probs = "candidate_probs" in df.columns
    for _, word_row in df.iterrows():
        guesses = word_row["candidates"].split()
        pos = word_row["pos"]
        real = real_lemma(word_row["word"], pos)
        values = {}

        # top 1 vs the real word
        height, _ancestor = lca([guesses[0], real], pos)
        values["lca_top1"] = height
        distance = glove_distance(guesses[0], real)
        values["glove_top1"] = np.nan if distance is None else distance
        values["emb_top1"] = embedding_distance(guesses[0], real, tok, E)

        for name, subset in [("top3", guesses[:3]), ("top5", guesses[:5])]:
            values[f"lca_{name}"], values[f"glove_{name}"], values[f"emb_{name}"] = spread_of(subset, pos, tok, E)

        if has_probs:
            probabilities = [float(p) for p in str(word_row["candidate_probs"]).split()]
            kept = likely_guesses(guesses, probabilities)
            values["n_likely"] = len(kept)
            values["lca_likely"], values["glove_likely"], values["emb_likely"] = spread_of(kept, pos, tok, E)
        else:
            for column in ["n_likely", "lca_likely", "glove_likely", "emb_likely"]:
                values[column] = np.nan
        rows.append(values)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ------------------------------------------------------------------------------------------
# distances between whole curves
# ------------------------------------------------------------------------------------------
def curve(df, column):
    """One text's raw curve for `column`; missing values (e.g. a word GloVe doesn't know) are dropped."""
    values = df[column].to_numpy(dtype=float)
    return values[~np.isnan(values)]


def standardise(values):
    spread = values.std()
    if spread == 0:
        return values - values.mean()
    return (values - values.mean()) / spread


def stretch(values, length=STRETCH_LENGTH):
    """Resample a curve to `length` points by linear interpolation over its relative position (0 = start, 1 = end)."""
    old_positions = np.linspace(0, 1, len(values))
    new_positions = np.linspace(0, 1, length)
    return np.interp(new_positions, old_positions, values)


def dtw_radius(curves):
    return max(1, int(round(WINDOW_SHARE * np.median([len(c) for c in curves]))))


def distance_table(curves, distance):
    """25 x 25 distances between curves: 'DTW' (window = 10% of the median length) or 'Euclidean' (stretched)."""
    if distance == "DTW":
        return cdist_dtw(to_time_series_dataset(curves), global_constraint="sakoe_chiba",
                         sakoe_chiba_radius=dtw_radius(curves))
    stretched = np.array([stretch(c) for c in curves])
    differences = stretched[:, None, :] - stretched[None, :, :]
    return np.sqrt((differences ** 2).sum(axis=-1))


# ------------------------------------------------------------------------------------------
# clustering
# ------------------------------------------------------------------------------------------
def k_distance_elbow(table, k=MIN_SAMPLES):
    """DBSCAN's distance limit: the elbow of the sorted distances of each text to its k-th nearest other text."""
    kth = []
    for row in table:
        others = np.sort(row)[1:]
        kth.append(others[k - 1])
    kth = np.sort(np.array(kth))
    positions = np.arange(len(kth))
    start = np.array([0, kth[0]])
    end = np.array([len(kth) - 1, kth[-1]])
    if np.allclose(end, start):
        return kth[0]
    line = (end - start) / np.linalg.norm(end - start)
    points = np.column_stack([positions, kth]) - start
    from_line = np.abs(points[:, 0] * line[1] - points[:, 1] * line[0])
    return kth[int(np.argmax(from_line))]


def k_medoids_pairs(table):
    """Exact k-medoids for k = 2: the pair of texts that, as the two centres, gives the smallest total distance."""
    best_cost, best_labels = None, None
    for a, b in combinations(range(len(table)), 2):
        nearest_is_b = table[:, b] < table[:, a]
        cost = np.where(nearest_is_b, table[:, b], table[:, a]).sum()
        if best_cost is None or cost < best_cost:
            best_cost, best_labels = cost, nearest_is_b.astype(int)
    return best_labels


def silhouette_or_nan(table, cluster_labels):
    keep = cluster_labels != -1
    if len(set(cluster_labels[keep])) < 2:
        return np.nan
    return silhouette_score(table[keep][:, keep], cluster_labels[keep], metric="precomputed")


def sorted_correctly(true_kinds, cluster_labels):
    """How many texts landed in a cluster whose majority kind is their own (DBSCAN noise = not sorted)."""
    true_kinds = np.asarray(true_kinds)
    correct = np.zeros(len(true_kinds), dtype=bool)
    for cluster in set(cluster_labels):
        if cluster == -1:
            continue
        members = cluster_labels == cluster
        majority = pd.Series(true_kinds[members]).mode()[0]
        correct[members] = true_kinds[members] == majority
    return correct


def cluster_curves(curves, true_kinds):
    """Run every scaling x distance x method on one set of curves. Returns a list of result dicts."""
    results = []
    for scaling in ["raw", "standardised"]:
        prepared = curves
        if scaling == "standardised":
            prepared = [standardise(c) for c in curves]
        for distance in ["DTW", "Euclidean"]:
            table = distance_table(prepared, distance)
            methods = {"DBSCAN": DBSCAN(eps=k_distance_elbow(table), min_samples=MIN_SAMPLES,
                                        metric="precomputed").fit_predict(table),
                       "k-medoids": k_medoids_pairs(table)}
            if distance == "Euclidean":
                stretched = np.array([stretch(c) for c in prepared])
                methods["KMeans"] = KMeans(n_clusters=K, n_init=50, random_state=0).fit_predict(stretched)
            else:
                dtw_kmeans = TimeSeriesKMeans(n_clusters=K, metric="dtw", random_state=0, n_init=3, max_iter=30,
                                              metric_params={"global_constraint": "sakoe_chiba",
                                                             "sakoe_chiba_radius": dtw_radius(prepared)})
                methods["DTW-KMeans"] = dtw_kmeans.fit_predict(to_time_series_dataset(prepared))
            for method, cluster_labels in methods.items():
                cluster_labels = np.asarray(cluster_labels)
                correct = sorted_correctly(true_kinds, cluster_labels)
                results.append({"scaling": scaling, "distance": distance, "method": method,
                                "ARI": adjusted_rand_score(true_kinds, cluster_labels),
                                "silhouette": silhouette_or_nan(table, cluster_labels),
                                "sorted correctly": int(correct.sum()),
                                "factual correct": int(correct[np.asarray(true_kinds) == "factual"].sum()),
                                "creative correct": int(correct[np.asarray(true_kinds) == "creative"].sum()),
                                "labels": cluster_labels})
    return results
