"""Is a signal's classification a robust result? Four tests, run in order (a signal that fails one is not
put through the later, slower ones):

    1. strong          its best setup sorts at least ROBUST_SHARE of the texts correctly
    2. beats chance    shuffling the texts' kinds at random and taking the best of the same 12 setups
                       reaches that score in fewer than CHANCE_LEVEL of CHANCE_SHUFFLES shuffles
    3. leave one out   the best setup, re-run with each text left out in turn, still sorts at least
                       ROBUST_SHARE correctly every time (so no single text carries the result)
    4. DTW window      the best setup still sorts at least ROBUST_SHARE correctly with every DTW window in
                       WINDOWS_TO_TEST (Euclidean setups have no window and pass automatically)
"""
import numpy as np

from analysis.classify import SETUPS, group, group_all_setups, score, setup_name, sorted_correctly
from config import CHANCE_LEVEL, CHANCE_SHUFFLES, RANDOM_SEED, ROBUST_SHARE, WINDOWS_TO_TEST


def best_setup(scores_per_setup):
    """The setup with the most texts sorted correctly (ties: the first in SETUPS order)."""
    best = None
    for setup in SETUPS:
        if best is None or scores_per_setup[setup]["correct"] > scores_per_setup[best]["correct"]:
            best = setup
    return best


def chance_scores(labels_per_setup, kinds, shuffles=CHANCE_SHUFFLES, seed=RANDOM_SEED):
    """For each random shuffle of the kinds: the best number sorted correctly over all setups.
    The groups don't depend on the kinds, so only the scoring is repeated."""
    generator = np.random.default_rng(seed)
    best_per_shuffle = []
    for _shuffle in range(shuffles):
        shuffled_kinds = generator.permutation(kinds)
        best = 0
        for labels in labels_per_setup.values():
            best = max(best, int(sorted_correctly(shuffled_kinds, labels).sum()))
        best_per_shuffle.append(best)
    return np.array(best_per_shuffle)


def leave_one_out(curves, kinds, names, setup):
    """Re-run `setup` once per text, with that text left out. One row per left-out text."""
    rows = []
    for left_out in range(len(curves)):
        kept_curves = []
        kept_kinds = []
        for index in range(len(curves)):
            if index != left_out:
                kept_curves.append(curves[index])
                kept_kinds.append(kinds[index])
        result = score(kept_kinds, group(kept_curves, setup))
        rows.append({"left out": names[left_out], "correct": result["correct"], "texts": result["texts"],
                     "share correct": result["share correct"]})
    return rows


def window_test(curves, kinds, setup):
    """Re-run `setup` with each DTW window in WINDOWS_TO_TEST. One row per window."""
    rows = []
    for window_share in WINDOWS_TO_TEST:
        result = score(kinds, group(curves, setup, window_share))
        rows.append({"window": window_share, "correct": result["correct"], "texts": result["texts"],
                     "share correct": result["share correct"]})
    return rows


def judge(signal, curves, kinds, names):
    """Run the four tests on one signal.

    Returns (verdict, setup_rows, leave_one_out_rows, window_rows):
        verdict             one dict summarising the signal (see the keys below)
        setup_rows          the score of every setup
        leave_one_out_rows  test 3 (empty if the signal failed earlier)
        window_rows         test 4 (empty if the signal failed earlier or the best setup has no DTW)
    """
    labels_per_setup = group_all_setups(curves)
    scores_per_setup = {}
    setup_rows = []
    for setup, labels in labels_per_setup.items():
        scores_per_setup[setup] = score(kinds, labels)
        row = {"signal": signal.key, "setup": setup_name(setup)}
        row.update(scores_per_setup[setup])
        setup_rows.append(row)

    best = best_setup(scores_per_setup)
    best_score = scores_per_setup[best]
    chance = chance_scores(labels_per_setup, kinds)
    chance_p = float((chance >= best_score["correct"]).mean())

    verdict = {"signal": signal.key, "name": signal.name, "family": signal.family, "time axis": signal.time_axis,
               "best setup": setup_name(best), "correct": best_score["correct"], "texts": best_score["texts"],
               "share correct": best_score["share correct"],
               "factual correct": best_score["factual correct"], "creative correct": best_score["creative correct"],
               "chance median": float(np.median(chance)), "chance 99th percentile": float(np.percentile(chance, 99)),
               "chance p": chance_p,
               "leave-one-out worst": np.nan, "leave-one-out median": np.nan, "window worst": np.nan}

    leave_one_out_rows = []
    window_rows = []
    if best_score["share correct"] < ROBUST_SHARE:
        verdict["failed test"] = "strong"
    elif chance_p >= CHANCE_LEVEL:
        verdict["failed test"] = "beats chance"
    else:
        leave_one_out_rows = leave_one_out(curves, kinds, names, best)
        shares = []
        for row in leave_one_out_rows:
            row["signal"] = signal.key
            shares.append(row["share correct"])
        verdict["leave-one-out worst"] = min(shares)
        verdict["leave-one-out median"] = float(np.median(shares))
        if best.distance == "DTW":
            window_rows = window_test(curves, kinds, best)
            window_shares = []
            for row in window_rows:
                row["signal"] = signal.key
                window_shares.append(row["share correct"])
            verdict["window worst"] = min(window_shares)
        if verdict["leave-one-out worst"] < ROBUST_SHARE:
            verdict["failed test"] = "leave one out"
        elif window_rows and verdict["window worst"] < ROBUST_SHARE:
            verdict["failed test"] = "DTW window"
        else:
            verdict["failed test"] = ""
    verdict["robust"] = verdict["failed test"] == ""
    return verdict, setup_rows, leave_one_out_rows, window_rows
