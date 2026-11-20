"""Analysis code for the Qwen7B notebooks.

    signals        what is measured at each word or token, and each text's curve of it
    cluster        DTW distances between raw curves, k-medoids into 2 clusters, the score, the text map
    figures        the result figures
    generate       writing a clean text and its copy with one hot burst (temperature, min_p)
    evolving       the evolving graph: distance to real factual / creative writing at every word
"""
import sys

from qwen_config import REPO_ROOT

# lca_core.py lives at the repo root; make it importable from here
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
