"""Bonus: can a simple SVM tell factual from creative text by the *shape* of the LCA curve?

Each (model, text) curve is cut into overlapping windows of content words; each
window becomes a feature vector (level, spread, jumpiness). Evaluation is
leave-one-text-out so windows of the same text never appear in train and test.

    python classify.py [results/k5_clean] [--pos n] [--metric lca_weighted]   # --pos n: nouns only (removes the
                                                # noun/verb-mix confound: verbs sit ~3 levels lower)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

WIN, STEP = 40, 20


def features(h):
    d = np.diff(h)
    return [h.mean(), h.var(), np.abs(d).mean(), np.abs(d).max(), (h >= 8).mean(),
            np.mean(np.sign(d[1:]) != np.sign(d[:-1]))]  # direction changes = convergence/divergence


def main(res_dir, pos=None, metric="lca_height"):
    X, y, groups = [], [], []
    for f in sorted(Path(res_dir).glob("*__*.csv")):
        text = f.stem.split("__")[1]
        df = pd.read_csv(f)
        h = df.loc[df["pos"] == pos if pos else slice(None), metric].to_numpy(float)
        for s in range(0, len(h) - WIN + 1, STEP):
            X.append(features(h[s:s + WIN]))
            y.append("factual" if text.startswith("factual") else "creative")
            groups.append(text)
    X, y = np.array(X), np.array(y)
    texts_per_class = pd.Series(dict(zip(groups, y))).value_counts()
    if len(texts_per_class) < 2 or texts_per_class.min() < 2:
        sys.exit(f"need >= 2 texts of each kind for leave-one-text-out, got {texts_per_class.to_dict()}")
    print(f"{len(X)} windows from {len(set(groups))} texts: {dict(zip(*np.unique(y, return_counts=True)))}")
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    pred = cross_val_predict(clf, X, y, groups=groups, cv=LeaveOneGroupOut())
    print(classification_report(y, pred, zero_division=0))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("res_dir", nargs="?", default=Path(__file__).parent / "results" / "k5_clean")
    ap.add_argument("--pos", choices=["n", "v", "a"])
    ap.add_argument("--metric", default="lca_height", choices=["lca_height", "lca_weighted"])
    a = ap.parse_args()
    main(a.res_dir, a.pos, a.metric)
