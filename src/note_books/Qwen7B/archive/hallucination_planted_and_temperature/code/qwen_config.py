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
# hallucination proof of concept (notebooks 03 and 04); kept apart from the real texts and their results
# ------------------------------------------------------------------------------------------
HALLUCINATED_DIR = HERE / "texts_hallucinated"
PLANTED_DIR = HALLUCINATED_DIR / "planted"      # copies with {{original => planted | type}} markers
GENERATED_DIR = HALLUCINATED_DIR / "generated"  # stories Qwen wrote (+ switch_points.csv, marks.csv)
HALLUCINATED_RUN_DIR = HERE / "results" / "run_hallucinated"
HALLUCINATED_WORDS_DIR = HALLUCINATED_RUN_DIR / "words"
HALLUCINATED_TOKENS_DIR = HALLUCINATED_RUN_DIR / "tokens"
HALLUCINATED_FIGURES_DIR = HERE / "results" / "hallucinated" / "figures"
HALLUCINATED_TABLES_DIR = HERE / "results" / "hallucinated" / "tables"

# writing the stories (Qwen itself writes them; temperature is the only sampling setting used)
STORY_PROMPTS = {
    "magic": "a girl in her first year at a school of magic who loses her wand the night before an important exam",
    "fairy": "a miller's son who is given three impossible tasks by a king before he may marry the princess",
    "detective": "a detective who works out, from small clues, who stole a necklace from a hotel room",
}
STORY_WORDS = 650                       # asked length, close to the median length of the 25 texts
GENERATION_MAX_TOKENS = 1000
LOW_TEMPERATURE = 0.7                   # the "clean" story
HIGH_TEMPERATURE = 1.0                  # the hallucination-prone story (1.15+ breaks Qwen into mixed scripts and code)
SWITCH_RANGE = (0.25, 0.75)             # the mid-run switch point is drawn at random in this share of the tokens

# temperature texts (notebooks 05-06): factual articles and stories, each cool, hot, and cool-then-hot (switch)
FACT_PROMPTS = {
    "paris": "Paris, the capital of France",
    "apollo": "the Apollo 11 Moon landing",
    "vaccines": "how vaccines train the immune system",
}
MIN_P = 0.1                             # drop tokens below 10% of the #1 token's probability (keeps hot text readable)
COOL_TEMPERATURE = 0.7                  # the "clean" text
HOT_TEMPERATURE = 2.6                   # the hallucinating text: the highest probed temperature still writing English, with wrong facts and invented places (notebook 05 section 2)
HOT_TEMPERATURES_TO_PROBE = [1.3, 1.5, 1.8, 2.2, 2.6]
TEMPERATURE_DIR = HALLUCINATED_DIR / "temperature"          # texts + switch_points.csv + marks.csv
TEMPERATURE_RUN_DIR = HERE / "results" / "run_temperature"
TEMPERATURE_WORDS_DIR = TEMPERATURE_RUN_DIR / "words"
TEMPERATURE_TOKENS_DIR = TEMPERATURE_RUN_DIR / "tokens"
TEMPERATURE_FIGURES_DIR = HERE / "results" / "temperature" / "figures"
TEMPERATURE_TABLES_DIR = HERE / "results" / "temperature" / "tables"
BURST_TEMPERATURE = 2.6                 # hot stretch inside a cool text (2.2 stayed correct: archive/burst_2.2)
BURST_TOKENS = 120                      # a few sentences: 40 tokens stayed correct (archive/burst_2.6_40tokens)
BURST_DIR = HALLUCINATED_DIR / "burst"                      # texts + bursts.csv
BURST_RUN_DIR = HERE / "results" / "run_burst"
BURST_WORDS_DIR = BURST_RUN_DIR / "words"
BURST_TOKENS_DIR = BURST_RUN_DIR / "tokens"
BURST_FIGURES_DIR = HERE / "results" / "burst" / "figures"
BURST_TABLES_DIR = HERE / "results" / "burst" / "tables"
EVOLVING_SIGNALS = ["glove_top5", "lca_likely_top3_10pct"]  # the two best factual/creative signals
EVOLVING_WINDOW = 15                    # the evolving graph compares the last 15 measured words with real writing

# analysis
PLACEMENT_SIGNAL = "glove_top5"         # cluster placement: the best factual/creative signal
WORD_LEVEL_SIGNALS = ["surprisal_per_word", "glove_top1"]   # word-by-word view: signals that see the real word
REACTION_WORDS = 5                      # a reaction is looked for at the planted word and the next 5 measured words
ANOMALY_WINDOW = 3                      # matrix profile: stretches of 3 measured words are compared
ANOMALIES_SHOWN = 3                     # the 3 most unusual (non-overlapping) stretches of each text are marked

# second reader: another model family reads the hallucinated texts (and the planted texts' clean originals), to see
# whether the word-level results depend on Qwen (and on Qwen reading stories it wrote itself)
SECOND_READER_MODEL = "allenai/OLMo-2-1124-7B-Instruct"
SECOND_READER_TOKENS_DIR = HERE / "results" / "run_second_reader" / "tokens"   # token tables only (one fast pass)
