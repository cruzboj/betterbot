"""The signals: what is measured at each word (or token) of a text, and how a text's curve is built from it.

Two families:

    guess meaning     how different the model's guesses for the next noun/verb are in meaning, measured three
                      ways (WordNet LCA height, GloVe distance, Qwen-embedding distance), each on four guess
                      sets (top 1 vs the real word, top 3, top 5, likely guesses only)
    model confidence  LM-Polygraph's token uncertainty values (surprisal, entropy, ...), read either at each
                      measured noun/verb or at every token of the text

To add a signal: put its value in a column of the word table (see add_guess_signals) or token table
(see lca_core.LCAMeter), then add one Signal(...) to SIGNALS. It then goes through the same classification
and tests as every other signal.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


from qwen_config import EXCLUDED_TEXTS, LIKELY_SHARE, TOKENS_DIR, WORDS_DIR
from lca_core import (POS, base_lemma, embedding_distance, embedding_spread, glove_distance, glove_spread, kind,
                      lca, load_token_embeddings)


@dataclass(frozen=True)
class Signal:
    key: str          # short id, also the column name in the word/token table
    name: str         # readable name used in figures and tables
    family: str       # "guess meaning" or "model confidence"
    time_axis: str    # "word" = one value per measured noun/verb, "token" = one value per token
    column: str       # the table column holding the value
    how: str          # one sentence: how the value is computed


# ------------------------------------------------------------------------------------------
# the list of signals
# ------------------------------------------------------------------------------------------
MEANING_MEASURES = {
    "lca": ("LCA height", "WordNet levels climbed until the words share a category (0 = same word)"),
    "glove": ("GloVe distance", "average cosine distance between the words in GloVe word vectors"),
    "emb": ("Qwen-embedding distance", "average cosine distance between the words in Qwen's own word vectors"),
}
GUESS_SETS = {
    "top1": ("top 1 vs real word", "between the model's #1 guess and the word that really came"),
    "top3": ("top 3 guesses", "between the model's 3 best guesses"),
    "top5": ("top 5 guesses", "between the model's 5 best guesses"),
    "likely": ("likely guesses", f"between the guesses with at least {LIKELY_SHARE:.0%} of the #1 guess's "
                                 "probability; 0 when only the #1 guess qualifies"),
}
CONFIDENCE_VALUES = {
    "surprisal": ("surprisal", "-log of the probability the model gave the real next token"),
    "max_prob": ("max probability", "probability of the model's #1 guess for the next token"),
    "entropy": ("entropy", "how spread out the model's whole next-token distribution is"),
    "self_certainty": ("self-certainty", "average log-probability over the whole vocabulary, plus log V"),
    "renyi": ("Rényi divergence", "Rényi divergence (alpha 0.5) between the next-token distribution "
                                  "(temperature 2) and a uniform one"),
    "fisher_rao": ("Fisher-Rao distance", "Fisher-Rao distance between the next-token distribution "
                                          "(temperature 2) and a uniform one"),
}


def _build_signal_list():
    signals = []
    for measure, (measure_name, measure_how) in MEANING_MEASURES.items():
        for guess_set, (set_name, set_how) in GUESS_SETS.items():
            key = f"{measure}_{guess_set}"
            signals.append(Signal(key=key, name=f"{measure_name}, {set_name}", family="guess meaning",
                                  time_axis="word", column=key, how=f"{measure_how}, {set_how}"))
    for time_axis in ["word", "token"]:
        for column, (value_name, value_how) in CONFIDENCE_VALUES.items():
            if time_axis == "word":
                where = "at the first token of each measured noun/verb"
            else:
                where = "at every token"
            signals.append(Signal(key=f"{column}_per_{time_axis}", name=f"{value_name} per {time_axis}",
                                  family="model confidence", time_axis=time_axis, column=column,
                                  how=f"{value_how}, {where}"))
    return signals


SIGNALS = _build_signal_list()


# ------------------------------------------------------------------------------------------
# guess-meaning values for one measured word
# ------------------------------------------------------------------------------------------
def real_lemma(word, pos):
    """Dictionary form of the real word, reduced the same way as the guesses ('hands' -> 'hand')."""
    lemma = base_lemma(word.lower(), POS[pos])
    if lemma is None:
        return word.lower()
    return lemma


def likely_guesses(guesses, probabilities):
    """The guesses whose probability is at least LIKELY_SHARE times the #1 guess's probability (best first)."""
    cutoff = LIKELY_SHARE * probabilities[0]
    kept = []
    for guess, probability in zip(guesses, probabilities):
        if probability >= cutoff:
            kept.append(guess)
    return kept


def meaning_spread(words, pos, tokenizer, embeddings):
    """(LCA height, GloVe distance, Qwen-embedding distance) of a set of words.
    A single word has no spread: all three are 0."""
    if len(words) < 2:
        return 0.0, 0.0, 0.0
    height, _ancestor = lca(words, pos)
    glove_mean, _glove_max = glove_spread(words)
    embedding_mean, _embedding_max = embedding_spread(words, tokenizer, embeddings)
    return height, glove_mean, embedding_mean


def guess_values(word_row, tokenizer, embeddings):
    """All 12 guess-meaning values (3 measures x 4 guess sets) for one row of a word table."""
    guesses = word_row["candidates"].split()
    probabilities = []
    for probability in str(word_row["candidate_probs"]).split():
        probabilities.append(float(probability))
    pos = word_row["pos"]
    values = {}

    # top 1: distance between the #1 guess and the real word
    real = real_lemma(word_row["word"], pos)
    height, _ancestor = lca([guesses[0], real], pos)
    values["lca_top1"] = height
    glove = glove_distance(guesses[0], real)
    if glove is None:
        glove = np.nan                      # GloVe doesn't know one of the two words
    values["glove_top1"] = glove
    values["emb_top1"] = embedding_distance(guesses[0], real, tokenizer, embeddings)

    # spreads of three guess sets
    guess_sets = {"top3": guesses[:3], "top5": guesses[:5], "likely": likely_guesses(guesses, probabilities)}
    for set_key, words in guess_sets.items():
        height, glove, embedding = meaning_spread(words, pos, tokenizer, embeddings)
        values[f"lca_{set_key}"] = height
        values[f"glove_{set_key}"] = glove
        values[f"emb_{set_key}"] = embedding
    values["likely_count"] = len(guess_sets["likely"])
    return values


def add_guess_signals(word_table, tokenizer, embeddings):
    """The word table with the 12 guess-meaning columns (and likely_count) added."""
    rows = []
    for _index, word_row in word_table.iterrows():
        rows.append(guess_values(word_row, tokenizer, embeddings))
    added = pd.DataFrame(rows)
    return pd.concat([word_table.reset_index(drop=True), added], axis=1)


# ------------------------------------------------------------------------------------------
# loading texts and building curves
# ------------------------------------------------------------------------------------------
def all_text_names():
    """Every text the model was run on (sorted file names without .csv)."""
    names = []
    for path in sorted(WORDS_DIR.glob("*.csv")):
        names.append(path.stem)
    return names


def dataset_names():
    """The texts used for the results: every text except those in EXCLUDED_TEXTS (qwen_config.py)."""
    kept = []
    for name in all_text_names():
        if name not in EXCLUDED_TEXTS:
            kept.append(name)
    return kept


def kinds_of(names):
    """'factual' / 'creative' for each text name, as a numpy array."""
    kinds = []
    for name in names:
        kinds.append(kind(name))
    return np.array(kinds)


def load_tables(names, model_name):
    """({name: word table with guess signals}, {name: token table}) for the given texts."""
    tokenizer, embeddings = load_token_embeddings(model_name)
    word_tables = {}
    token_tables = {}
    for name in names:
        word_table = pd.read_csv(WORDS_DIR / f"{name}.csv")
        word_tables[name] = add_guess_signals(word_table, tokenizer, embeddings)
        token_tables[name] = pd.read_csv(TOKENS_DIR / f"{name}.csv")
    return word_tables, token_tables


def curve(table, column):
    """One text's curve: the column's values in reading order, skipping missing values
    (e.g. a word GloVe doesn't know)."""
    values = table[column].to_numpy(dtype=float)
    return values[~np.isnan(values)]


def curves_of(signal, names, word_tables, token_tables):
    """The curve of `signal` for each text in `names` (same order)."""
    if signal.time_axis == "word":
        tables = word_tables
    else:
        tables = token_tables
    curves = []
    for name in names:
        curves.append(curve(tables[name], signal.column))
    return curves
