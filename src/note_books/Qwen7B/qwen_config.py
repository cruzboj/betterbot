"""Every setting of the Qwen7B pipeline, in one place.

Change a value here and re-run the notebooks; nothing else hard-codes these numbers.
"""
from pathlib import Path

# ------------------------------------------------------------------------------------------
# folders
# ------------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # the Qwen7B folder
REPO_ROOT = HERE.parents[2]                     # the folder holding lca_core.py
TEXTS_DIR = HERE / "texts"                      # one .txt per text; the name starts with its kind
RUN_DIR = HERE / "results" / "run"              # what notebook 01 saves (model output per text)
WORDS_DIR = RUN_DIR / "words"                   # one row per measured noun/verb
TOKENS_DIR = RUN_DIR / "tokens"                 # one row per token
FIGURES_DIR = HERE / "results" / "figures"      # what notebook 02 saves
TABLES_DIR = HERE / "results" / "tables"

# ------------------------------------------------------------------------------------------
# model run (notebook 01)
# ------------------------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LOAD_IN_4BIT = True             # 4-bit weights so the 7B model fits an 8 GB GPU
GUESSES_PER_WORD = 5            # the model's top guesses kept at each noun/verb
WHOLE_WORD_GUESSES = True       # keep only guesses that are complete words

# texts left out of the results by decision: {"text name": "reason"}. Empty: all texts are used (docs/adr/0002)
EXCLUDED_TEXTS = {}

# ------------------------------------------------------------------------------------------
# signals
# ------------------------------------------------------------------------------------------
LIKELY_SHARE = 0.10             # a "likely" guess has at least 10% of the #1 guess's probability

# ------------------------------------------------------------------------------------------
# clustering (k-medoids into 2 clusters on DTW distances between raw curves)
# ------------------------------------------------------------------------------------------
DTW_WINDOW_SHARE = 0.10         # DTW may shift a curve by at most 10% of the median curve length
MAPS_TO_DRAW = 10               # text maps are drawn for the signals that cluster best
RANDOM_SEED = 0                 # only for the layout of the text map
