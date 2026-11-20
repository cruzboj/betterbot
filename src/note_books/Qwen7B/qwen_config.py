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
LIKELY_SHARE_SWEEP = [0.05, 0.10, 0.20]     # shares tried for the sweep signals (likely, at most top 3; top 1 if likely)

# ------------------------------------------------------------------------------------------
# clustering (k-medoids into 2 clusters on DTW distances between raw curves)
# ------------------------------------------------------------------------------------------
DTW_WINDOW_SHARE = 0.10         # DTW may shift a curve by at most 10% of the median curve length
MAPS_TO_DRAW = 10               # text maps are drawn for the signals that cluster best
RANDOM_SEED = 0                 # only for the layout of the text map

# ------------------------------------------------------------------------------------------
# hallucination burst (notebook 03): clean texts Qwen writes, and copies with one hot stretch
# ------------------------------------------------------------------------------------------
FACT_PROMPTS = {
    "paris": "Paris, the capital of France",
    "apollo": "the Apollo 11 Moon landing",
    "vaccines": "how vaccines train the immune system",
}
STORY_PROMPTS = {
    "magic": "a girl in her first year at a school of magic who loses her wand the night before an important exam",
    "fairy": "a miller's son who is given three impossible tasks by a king before he may marry the princess",
    "detective": "a detective who works out, from small clues, who stole a necklace from a hotel room",
}
STORY_WORDS = 650                       # asked length, close to the median length of the 25 texts
GENERATION_MAX_TOKENS = 1000
MIN_P = 0.1                             # drop tokens below 10% of the #1 token's probability (keeps hot text readable)
COOL_TEMPERATURE = 0.7                  # the clean text
BURST_TEMPERATURE = 3.0                 # the hot stretch: highest that stays readable (2.2 and 2.6 gave fewer wrong facts, 3.5+ word salad: archive/burst_trials)
BURST_TOKENS = 120                      # a few sentences (40 tokens stayed correct: archive/burst_trials)
BURST_RANGE = (0.25, 0.75)              # the burst starts at a random token in this share of the clean text

BURST_DIR = HERE / "texts_hallucinated" / "burst"     # clean + burst texts, bursts.csv
BURST_RUN_DIR = HERE / "results" / "run_burst"        # what Qwen measured reading them
BURST_WORDS_DIR = BURST_RUN_DIR / "words"
BURST_TOKENS_DIR = BURST_RUN_DIR / "tokens"
BURST_FIGURES_DIR = HERE / "results" / "burst" / "figures"
BURST_TABLES_DIR = HERE / "results" / "burst" / "tables"

# the evolving graph: the two best factual/creative signals of notebook 02, plus surprisal per word (how unexpected
# each real word was to the model), compared with real writing word by word
EVOLVING_SIGNALS = ["glove_top5", "lca_likely_top3_10pct", "surprisal_per_word"]
EVOLVING_WINDOW = 15                    # the last 15 measured words are compared with real writing
