"""The evolving graph: how a text's curve compares with real writing while it is being read (notebook 06).

    reference writing    the 25 real texts, split into the two fixed clusters (docs/adr/0003), each cluster named
                         after the kind most of its texts have ("factual" / "creative")
    distance to writing  at every word: the last `window` words of the raw curve against the most similar stretch
                         of the same length anywhere in a cluster's texts (plain Euclidean on raw values, a
                         matrix-profile AB-join: stumpy.aamp); low = this stretch reads like that kind of writing
    normal ceiling       the highest distance to the nearest writing that any real text reaches, each real text
                         compared without itself (a maximum, nothing averaged)
    alarm                the first word where a text goes above the normal ceiling
"""
import warnings

import numpy as np
import pandas as pd

from analysis.cluster import cluster


def references_by_cluster(names, kinds, curves):
    """{"factual": {name: curve}, "creative": {name: curve}}: the real texts grouped by the k-medoids cluster they
    fall in, each cluster named after the kind most of its texts have."""
    labels, _centres, _table = cluster(curves)
    kinds = np.asarray(kinds)
    references = {}
    for cluster_label in sorted(set(labels)):
        members = np.flatnonzero(labels == cluster_label)
        majority_kind = pd.Series(kinds[members]).mode()[0]
        if majority_kind in references:
            raise ValueError(f"both clusters are mostly {majority_kind}: they cannot be told apart")
        references[majority_kind] = {}
        for index in members:
            references[majority_kind][names[index]] = curves[index]
    return references


def distance_to_writing(curve, references, window, leave_out=()):
    """At every word t: the distance between the stretch curve[t-window+1 : t+1] and the most similar stretch in
    any reference curve (NaN before the first full stretch, and everywhere when the curve is shorter than the
    window). References shorter than the window, or named in `leave_out`, are skipped."""
    import stumpy
    curve = np.asarray(curve, dtype=float)
    kept = []
    for name, reference in references.items():
        if name in leave_out or len(reference) < window:
            continue
        kept.append(np.asarray(reference, dtype=float))
    if len(kept) == 0:
        raise ValueError("no reference curve left to compare with")
    distances = np.full(len(curve), np.nan)
    if len(curve) < window:
        return distances
    best = None
    for reference in kept:
        with warnings.catch_warnings():          # stumpy warns when a reference equals the curve: that is fine here
            warnings.simplefilter("ignore", UserWarning)
            profile = stumpy.aamp(curve, window, reference, ignore_trivial=False)[:, 0].astype(float)
        if best is None:
            best = profile
        else:
            best = np.minimum(best, profile)
    distances[window - 1:] = best        # profile[i] is the stretch starting at i, i.e. ending at i + window - 1
    return distances


def evolving_profile(curve, references_by_kind, window, leave_out=()):
    """One text's evolving graph as a table: word, value (the raw curve), to factual, to creative, to nearest.
    A kind with no reference left after `leave_out` gets an all-NaN column."""
    curve = np.asarray(curve, dtype=float)
    table = pd.DataFrame({"word": np.arange(len(curve)), "value": curve})
    for kind in ["factual", "creative"]:
        try:
            table[f"to {kind}"] = distance_to_writing(curve, references_by_kind[kind], window, leave_out)
        except ValueError:
            table[f"to {kind}"] = np.nan
    table["to nearest"] = np.fmin(table["to factual"].to_numpy(), table["to creative"].to_numpy())
    return table


def normal_ceiling(real_curves, references_by_kind, window):
    """The highest distance to the nearest writing any real text reaches when it is left out of the references."""
    ceiling = 0.0
    for name, values in real_curves.items():
        nearest = evolving_profile(values, references_by_kind, window, leave_out=(name,))["to nearest"]
        if nearest.notna().any():
            ceiling = max(ceiling, float(nearest.max()))
    return ceiling


def first_alarm(to_nearest, ceiling):
    """The first word whose distance to the nearest writing is above `ceiling` (NaN never is), or None."""
    above = np.flatnonzero(np.nan_to_num(np.asarray(to_nearest, dtype=float), nan=-np.inf) > ceiling)
    if len(above) == 0:
        return None
    return int(above[0])
