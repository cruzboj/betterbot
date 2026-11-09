"""Shared LCA toolkit for the betterbot notebooks (lives at the repo root).

# ---- copy this into the first cell of any notebook / script in the repo -------------
# import sys
# from pathlib import Path
#
# # walk up from the current folder until we find the repo root (the folder holding lca_core.py)
# ROOT = Path.cwd()
# while not (ROOT / "lca_core.py").exists():
#     if ROOT == ROOT.parent:
#         raise FileNotFoundError("lca_core.py not found in this folder or any parent folder")
#     ROOT = ROOT.parent
# sys.path.insert(0, str(ROOT))          # makes lca_core importable from any sub-folder
#
# from lca_core import (
#     # WordNet LCA
#     lca, lca_height, name_of, weighted_pair_height, sanity_check,
#     # pairwise helpers + tree drawing
#     get_synset, distance_to_lca, build_lca_graph, draw_lca_graph,
#     # measurement points
#     content_words, candidate_lemma, base_lemma, tag_to_pos,
#     # LLM measurement (loads torch/transformers on first use)
#     LCAMeter,
#     # input texts
#     load_texts, fetch_wiki, fetch_jokes, fetch_gutenberg, gutenberg_excerpt, clean_pdf_text, hp_excerpt,
#     # results + stats
#     kind, slug, load_results, summarize, curve_features, CURVE_FEATURE_NAMES,
#     windows, feature_table, pos_normalize,
#     # embedding spread (alternative to WordNet LCA)
#     load_token_embeddings, cosine_distance, embedding_distance, embedding_spread, add_embedding_spread,
#     load_glove, glove_distance, glove_spread, glove_log_rank, add_glove_spread,
# )
# -------------------------------------------------------------------------------------
# (From the repo root itself, plain `from lca_core import ...` is enough.)

Sections
    1. WordNet LCA          lca, lca_height, weighted_pair_height, sanity_check
    2. Pairwise helpers     get_synset, distance_to_lca, build_lca_graph, draw_lca_graph
    3. Measurement points   base_lemma, candidate_lemma, tag_to_pos, content_words
    4. LLM measurement      LCAMeter (needs torch + transformers, imported on first use)
    5. Input texts          fetch_wiki, fetch_jokes, fetch_gutenberg, gutenberg_excerpt, clean_pdf_text,
                            hp_excerpt, load_texts
    6. Results + stats      kind, slug, load_results, summarize, curve_features, windows, feature_table,
                            pos_normalize
    7. Embedding spread     load_token_embeddings, cosine_distance, embedding_distance, embedding_spread,
                            add_embedding_spread, load_glove, glove_distance, glove_spread, glove_log_rank,
                            add_glove_spread

Heavy libraries (torch, transformers, safetensors, networkx, pypdf) are imported only inside the
functions that need them, so `import lca_core` is fast and works without a GPU.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn

for _resource in ["wordnet", "stopwords", "averaged_perceptron_tagger_eng", "punkt_tab"]:
    nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords  # noqa: E402  (needs the download above)

__all__ = [
    "POS", "VIRTUAL_ROOT", "FILLERS", "STOP", "AUX_LEMMAS",
    "senses", "lca", "lca_height", "name_of", "weighted_pair_height", "sanity_check",
    "get_synset", "distance_to_lca", "build_lca_graph", "draw_lca_graph",
    "base_lemma", "candidate_lemma", "tag_to_pos", "content_words",
    "LCAMeter", "TOKEN_STAT_COLUMNS",
    "fetch_wiki", "fetch_jokes", "fetch_gutenberg", "gutenberg_excerpt", "first_words", "clean_pdf_text",
    "hp_excerpt", "load_texts",
    "kind", "slug", "load_results", "summarize", "curve_features", "CURVE_FEATURE_NAMES", "windows",
    "feature_table", "pos_normalize",
    "unslug", "load_token_embeddings", "cosine_distance", "word_vector", "embedding_distance",
    "embedding_spread", "add_embedding_spread",
    "load_glove", "glove_distance", "glove_spread", "glove_log_rank", "add_glove_spread",
]


# =====================================================================================
# 1. WordNet LCA
# =====================================================================================
POS = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ}
VIRTUAL_ROOT = "<ROOT>"  # verbs/adjectives have many roots; join them here


def senses(word, pos, max_senses=2):
    """The word's first `max_senses` senses (WordNet orders them by frequency)."""
    all_senses = wn.synsets(word, pos=pos)
    return all_senses[:max_senses]


def _parents(synset):
    """The synsets directly above `synset` in WordNet (normal and 'instance of' hypernyms)."""
    return synset.hypernyms() + synset.instance_hypernyms()


@lru_cache(maxsize=200_000)
def _up_distances(synset):
    """Shortest is-a distance from `synset` to each of its ancestors (incl. itself).

    Breadth-first search upwards: level 0 is the synset, level 1 its parents, and so on.
    The first time we reach an ancestor is along the shortest path.
    """
    dist = {synset: 0}
    current_level = [synset]
    while current_level:
        next_level = []
        for s in current_level:
            for parent in _parents(s):
                if parent not in dist:
                    dist[parent] = dist[s] + 1
                    next_level.append(parent)
        current_level = next_level

    # the top of WordNet: ancestors with no parents of their own
    roots = []
    for ancestor in dist:
        if not _parents(ancestor):
            roots.append(ancestor)

    # one extra level above the closest root joins all trees (verbs have many separate roots)
    closest_root_distance = None
    for root in roots:
        if closest_root_distance is None or dist[root] < closest_root_distance:
            closest_root_distance = dist[root]
    dist[VIRTUAL_ROOT] = closest_root_distance + 1
    return dist


def _word_distances(word, pos, max_senses):
    """ancestor -> shortest distance to it from any of the word's allowed senses."""
    shortest = {}
    for sense in senses(word, pos, max_senses):
        for ancestor, distance in _up_distances(sense).items():
            if ancestor not in shortest or distance < shortest[ancestor]:
                shortest[ancestor] = distance
    return shortest


def lca(words, pos="n", max_senses=2):
    """(height, lca_synset) for a set of words, or (None, None) if a word isn't in WordNet for this POS.

    height = min over common ancestors A of max_{w in words} dist(w, A): how many is-a levels the
    farthest word climbs until all words meet. Distances are minimised over each word's
    `max_senses` most frequent senses.
    """
    # drop repeated words, keeping their order
    unique_words = []
    for w in words:
        if w not in unique_words:
            unique_words.append(w)

    # for each word: {ancestor: distance}
    dists = []
    for w in unique_words:
        dists.append(_word_distances(w, POS[pos], max_senses))
    if not dists:
        return None, None
    for d in dists:
        if not d:  # this word isn't in WordNet as this part of speech
            return None, None

    # ancestors shared by every word
    common = set(dists[0]).intersection(*dists[1:])

    # pick the shared ancestor whose farthest word is closest; on a tie prefer a real synset over VIRTUAL_ROOT
    best, best_key = None, None
    for ancestor in common:
        farthest = 0
        for d in dists:
            farthest = max(farthest, d[ancestor])
        key = (farthest, ancestor == VIRTUAL_ROOT)
        if best_key is None or key < best_key:
            best, best_key = ancestor, key

    height = 0
    for d in dists:
        height = max(height, d[best])
    return height, best


def lca_height(*words, pos="n", max_senses=2):
    """Just the height: lca_height("dog", "cat") -> 2."""
    height, _ancestor = lca(words, pos, max_senses)
    return height


def name_of(ancestor):
    """Printable name of an LCA: the synset's name, or VIRTUAL_ROOT / None unchanged."""
    if ancestor is None or isinstance(ancestor, str):
        return ancestor
    return ancestor.name()


def weighted_pair_height(cands, pos="n", max_senses=2):
    """Expected LCA height of two different completions drawn from the model's distribution.

    `cands` is {word: probability}. Returns sum_{i<j} p_i p_j h(i,j) / sum_{i<j} p_i p_j, so a
    low-probability outlier only enters through its few pairs instead of pulling the whole set to
    the root as in the max-based `lca`.
    """
    words = list(cands.keys())
    probs = list(cands.values())
    weighted_sum = 0.0
    total_weight = 0.0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):  # every pair once
            weight = probs[i] * probs[j]
            pair_height, _ancestor = lca([words[i], words[j]], pos, max_senses)
            weighted_sum += weight * pair_height
            total_weight += weight
    return weighted_sum / total_weight


SANITY = [("dog", "cat", 1), ("canine", "cat", 2), ("dog", "chair", 8)]


def sanity_check(max_senses=2):
    """Print the dog/cat, canine/cat, dog/chair check from the 15.9 meeting."""
    print(f"{'pair':<16}{'estimate':>9}{'got':>5}  LCA")
    for a, b, estimate in SANITY:
        height, ancestor = lca([a, b], "n", max_senses)
        pair = a + "/" + b
        print(f"{pair:<16}{estimate:>9}{height:>5}  {name_of(ancestor)}")


# =====================================================================================
# 2. Pairwise helpers + tree drawing (from LCA.ipynb)
# =====================================================================================
def get_synset(term):
    """'dog' -> first sense of dog; 'canine.n.02' -> that exact synset."""
    if "." in term:
        return wn.synset(term)
    return wn.synsets(term)[0]


def distance_to_lca(synset1, synset2):
    """NLTK-style distance from synset1 up to its lowest common hypernym with synset2."""
    if isinstance(synset1, str):
        synset1 = get_synset(synset1)
    if isinstance(synset2, str):
        synset2 = get_synset(synset2)
    ancestor = synset1.lowest_common_hypernyms(synset2)[0]
    return synset1.shortest_path_distance(ancestor)


def _add_path(graph, synset, ancestor):
    """Add the is-a edges from `ancestor` down to `synset` (first hypernym path that contains it)."""
    for path in synset.hypernym_paths():  # each path runs from the top of WordNet down to synset
        if ancestor in path:
            segment = path[path.index(ancestor):]
            for i in range(len(segment) - 1):
                parent, child = segment[i], segment[i + 1]
                graph.add_edge(parent.name(), child.name())
            break


def build_lca_graph(pairs, graph=None):
    """networkx DiGraph of the is-a paths from each pair's words up to their LCA.

    pairs: [("dog", "cat"), ("dog", "chair"), ("dog", "canine.n.02"), ...]
    """
    import networkx as nx

    if graph is None:
        graph = nx.DiGraph()
    for word1, word2 in pairs:
        synset1 = get_synset(word1)
        synset2 = get_synset(word2)
        ancestor = synset1.lowest_common_hypernyms(synset2)[0]
        _add_path(graph, synset1, ancestor)
        _add_path(graph, synset2, ancestor)
    return graph


def draw_lca_graph(graph, out=None, figsize=(14, 8)):
    """Draw a graph from `build_lca_graph` as a left-to-right tree (uses graphviz if installed)."""
    import matplotlib.pyplot as plt
    import networkx as nx

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        graph.graph["rankdir"] = "LR"
        positions = graphviz_layout(graph, prog="dot")
    except Exception:  # no graphviz binary: fall back to layers by depth
        roots = []
        for node, n_parents in graph.in_degree():
            if n_parents == 0:
                roots.append(node)
        depth = {}
        for root in roots:
            for node, d in nx.single_source_shortest_path_length(graph, root).items():
                if node not in depth or d < depth[node]:
                    depth[node] = d
        nx.set_node_attributes(graph, depth, "layer")
        positions = nx.multipartite_layout(graph, subset_key="layer")
    plt.figure(figsize=figsize)
    nx.draw(graph, positions, with_labels=True, node_color="lightblue", node_size=300,
            font_size=8, arrows=True, arrowsize=10, edge_color="gray")
    plt.margins(0.05)
    if out:
        plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()


# =====================================================================================
# 3. Where to measure: nouns and verbs without fillers
# =====================================================================================
FILLERS = {"also", "being", "however", "thus", "therefore", "would", "could", "should",
           "may", "might", "must", "shall", "said", "say", "says", "get", "got", "make",
           "made", "go", "went", "one", "two", "like", "well", "even", "much", "many", "us"}
STOP = set(stopwords.words("english")) | FILLERS
AUX_LEMMAS = {"be", "have", "do"}


def base_lemma(word, pos):
    """WordNet lemma, preferring the base form ('hands' -> 'hand', 'saw' -> 'see')."""
    forms = wn._morphy(word, pos)  # every dictionary form WordNet can map `word` to
    if not forms:
        return None
    # prefer a form that differs from the word itself (the word may be an inflection, e.g. 'saw' -> 'see')
    other_forms = []
    for form in forms:
        if form != word:
            other_forms.append(form)
    if other_forms:
        options = other_forms
    else:
        options = forms
    # the shortest option (first one on a tie)
    shortest = options[0]
    for form in options[1:]:
        if len(form) < len(shortest):
            shortest = form
    return shortest


def _sense_count(word, pos):
    """How often WordNet's corpus counts used `word` as this part of speech (-1 if not a word of that POS)."""
    lemma = base_lemma(word, pos)
    if lemma is None:
        return -1
    total = 0
    for synset in wn.synsets(lemma, pos):
        for lemma_entry in synset.lemmas():
            if lemma_entry.name().lower() == lemma:
                total += lemma_entry.count()
    return total


@lru_cache(maxsize=None)
def candidate_lemma(word, pos):
    """Lemma of a guess if `pos` ('n'/'v'/'a') is its dominant usage in WordNet, else None."""
    lemma = base_lemma(word, POS[pos])
    if lemma is None:
        return None
    if pos == "v" and lemma in AUX_LEMMAS:
        return None

    count_as_this_pos = _sense_count(word, POS[pos])
    count_as_other_pos = -1
    for other_pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
        if other_pos != POS[pos]:
            count_as_other_pos = max(count_as_other_pos, _sense_count(word, other_pos))

    if count_as_this_pos >= count_as_other_pos:
        return lemma
    return None


def tag_to_pos(tag, use_adj=False):
    """Penn Treebank tag -> 'n' / 'v' / 'a' / None (proper nouns are skipped)."""
    is_noun = tag.startswith("NN") and not tag.startswith("NNP")
    if is_noun:
        return "n"
    if tag.startswith("VB"):
        return "v"
    if use_adj and tag.startswith("JJ"):
        return "a"
    return None


def content_words(text, use_adj=False):
    """[(char_start, word, pos)] of the measurement points in `text`."""
    points = []
    search_from = 0
    for sentence in nltk.sent_tokenize(text):
        sentence_start = text.find(sentence, search_from)
        search_from = sentence_start + len(sentence)

        # alphabetic words of the sentence, with their position in the full text
        positions, words = [], []
        for match in re.finditer(r"[A-Za-z]+", sentence):
            positions.append(sentence_start + match.start())
            words.append(match.group())
        tags = nltk.pos_tag(words)  # [(word, tag)] in the same order

        for i in range(len(words)):
            word, tag = words[i], tags[i][1]
            pos = tag_to_pos(tag, use_adj)
            if pos is None or word.lower() in STOP:
                continue
            lemma = base_lemma(word.lower(), POS[pos])
            if lemma is None:
                continue
            if pos == "v" and lemma in AUX_LEMMAS:
                continue
            points.append((positions[i], word, pos))
    return points


# =====================================================================================
# 4. Measuring a text with an LLM
# =====================================================================================
WORD_START_MARKERS = ("Ġ", "▁")      # how GPT-2/Pythia (Ġ) and SentencePiece (▁) tokens mark a leading space
NEWLINE_MARKER = "Ċ"                 # GPT-2-style newline token


TOKEN_STAT_COLUMNS = ["surprisal", "max_prob", "entropy", "self_certainty", "renyi", "fisher_rao"]


class LCAMeter:
    """Run a causal LM over a text and score the LCA of its top-k guesses at each content word.

        meter = LCAMeter("gpt2-medium")                   # loads the model itself
        meter = LCAMeter(model=model, tokenizer=tok)      # or use an already-loaded model (e.g. 4-bit)
        df = meter.run(text, k=5)      # one row per measurement point
        tokens = meter.last_tokens     # one row per token of the same text (uncertainty values)
        meter.free()

    Besides the LCA of the guesses, every token gets LM-Polygraph's token-level uncertainty values
    (TOKEN_STAT_COLUMNS, higher = more uncertain), computed from the same single pass over the text.
    """

    def __init__(self, model_name=None, device=None, model=None, tokenizer=None, chunk_size=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.chunk_size = chunk_size   # positions turned into full-vocabulary probabilities at once (memory)
        if model is not None:
            # already loaded (e.g. by src/model/load_model.py, possibly in 4-bit): use it as it is
            self.model = model.eval()
            self.tok = tokenizer
            self.device = str(model.device)
        else:
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            self.device = device

            self.tok = AutoTokenizer.from_pretrained(model_name)
            if self.device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
            loaded = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            self.model = loaded.to(self.device).eval()

        # how many tokens the model reads at once (its maximum, capped at 2048 to save memory)
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if not max_positions:
            max_positions = 1024
        self.window = min(max_positions, 2048)
        self.last_tokens = None
        self._build_vocab_tables()

    def _build_vocab_tables(self):
        """vocab_word: token id -> lowercase word, for tokens that start a new alphabetic word.
        boundary: tokens that END the current word when they come next (new word, punctuation, newline, EOS)."""
        torch = self.torch
        vocab_size = self.model.get_output_embeddings().weight.shape[0]
        self.vocab_word = {}
        self.boundary = torch.zeros(vocab_size, dtype=torch.bool)

        all_ids = list(range(len(self.tok)))
        for token_id, piece in enumerate(self.tok.convert_ids_to_tokens(all_ids)):
            if not isinstance(piece, str):
                continue
            text = self.tok.convert_tokens_to_string([piece])
            first_char = piece[:1]
            starts_new_word = first_char in WORD_START_MARKERS
            is_newline = first_char == NEWLINE_MARKER
            starts_with_punctuation = text != "" and not text[0].isalnum()
            self.boundary[token_id] = starts_new_word or is_newline or starts_with_punctuation

            word = text.strip()
            if starts_new_word and word.isalpha() and word.lower() not in STOP:
                self.vocab_word[token_id] = word.lower()

        if self.tok.eos_token_id is not None:
            self.boundary[self.tok.eos_token_id] = True
        self.boundary = self.boundary.to(self.device)

    def _completes_word(self, seq, j, token_ids, ctx=128):
        """For each candidate token at position j: is P(the next token ends the word) >= 0.5 ?"""
        torch = self.torch
        with torch.no_grad():
            context_start = max(0, j - ctx)
            context = seq[context_start:j]
            batch = torch.tensor([context + [t] for t in token_ids], device=self.device)

            # logits_to_keep=1: only the last position's scores (full logits over Qwen's vocab don't fit 8 GB)
            output = self.model(batch, logits_to_keep=1)
            last_position_logits = output.logits[:, -1].float()
            next_token_probs = torch.softmax(last_position_logits, dim=-1)

            prob_word_ends = next_token_probs[:, self.boundary].sum(dim=-1)
            return (prob_word_ends >= 0.5).tolist()

    def _token_stats(self, log_probs, true_ids):
        """LM-Polygraph's token-level uncertainty values for a block of positions (higher = more uncertain).

        log_probs: (positions, vocabulary) log-probabilities of the next token; true_ids: the real next tokens.
        Formulas follow lm_polygraph 0.7.0 (estimators/ and stat_calculators/entropy.py):
          surprisal      = -log p(real token)                      MaximumTokenProbability
          entropy        = -sum_v p_v log p_v                       TokenEntropy (EntropyCalculator)
          self_certainty = mean_v(log p_v) + log V                  per-token term of SelfCertainty (sign as the estimator)
          renyi          = Renyi divergence to uniform, alpha=0.5, T=2   per-token term of RenyiNeg
          fisher_rao     = Fisher-Rao distance to uniform, T=2      per-token term of FisherRao
        """
        torch = self.torch
        vocab_size = log_probs.shape[-1]
        log_vocab = torch.log(torch.tensor(float(vocab_size), device=log_probs.device))
        probs = log_probs.exp()
        rows = torch.arange(len(true_ids), device=log_probs.device)

        stats = {}
        stats["surprisal"] = -log_probs[rows, true_ids]
        stats["max_prob"] = probs.max(dim=-1).values
        stats["entropy"] = -(probs * log_probs).sum(dim=-1)
        stats["self_certainty"] = log_probs.mean(dim=-1) + log_vocab

        temperature, alpha = 2.0, 0.5
        q = torch.softmax(log_probs / temperature, dim=-1)       # flattened distribution used by RenyiNeg / FisherRao
        stats["renyi"] = (torch.log((q ** alpha).sum(dim=-1)) - (alpha - 1) * log_vocab) / (alpha - 1)
        overlap_with_uniform = q.sqrt().sum(dim=-1) * torch.sqrt(1 / torch.tensor(float(vocab_size)))
        stats["fisher_rao"] = 2 / torch.pi * torch.arccos(overlap_with_uniform.clamp(max=1.0))
        return stats

    def _scores_float32(self, hidden, output_layer, vocab_slice=16384):
        """The output layer (hidden state -> one score per vocabulary token) computed in 32-bit precision.

        Half-precision scores are rounded coarsely (bfloat16 steps are 0.125 for scores around 20), which
        visibly shifts probabilities. Converting the whole output layer to 32 bits would need ~2 GB for
        Qwen, so it is done one slice of the vocabulary at a time.
        """
        torch = self.torch
        hidden = hidden.float()
        weight = output_layer.weight          # (vocabulary, hidden size)
        bias = output_layer.bias
        slices = []
        for first in range(0, weight.shape[0], vocab_slice):
            last = min(first + vocab_slice, weight.shape[0])
            scores = hidden @ weight[first:last].float().T
            if bias is not None:
                scores = scores + bias[first:last].float()
            slices.append(scores)
        return torch.cat(slices, dim=-1)

    def _predictions(self, seq, top_n):
        """For every position j>=1: top_n (probs, ids) of P(token_j | seq[:j]), P(actual token), and the
        token-level uncertainty values (see _token_stats) of that prediction.

        Long texts are read in overlapping windows; each new window keeps at least half a window
        of earlier text as context. Within a window, full-vocabulary probabilities are computed
        chunk_size positions at a time, so large vocabularies (Qwen: 152k) fit in GPU memory.
        """
        torch = self.torch
        base_model = self.model.base_model                 # the model without its output layer
        output_layer = self.model.get_output_embeddings()  # hidden state -> one score per vocabulary token
        with torch.no_grad():
            n = len(seq)
            stride = self.window // 2
            top_p = torch.full((n, top_n), float("nan"))
            top_i = torch.zeros((n, top_n), dtype=torch.long)
            actual = torch.full((n,), float("nan"))
            token_stats = {}
            for name in TOKEN_STAT_COLUMNS:
                token_stats[name] = torch.full((n,), float("nan"))

            done = 1  # positions [1, done) already have predictions (position 0 has no context)
            while done < n:
                if done == 1:
                    end = min(self.window, n)
                else:
                    end = min(done + stride, n)
                start = max(0, end - self.window)

                window_ids = torch.tensor([seq[start:end]], device=self.device)
                hidden = base_model(window_ids).last_hidden_state[0]

                # the prediction for position p is made at position p - 1 (in window coordinates: p - 1 - start)
                for block_start in range(done, end, self.chunk_size):
                    block_end = min(block_start + self.chunk_size, end)
                    block_hidden = hidden[block_start - 1 - start:block_end - 1 - start]
                    logits = self._scores_float32(block_hidden, output_layer)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()

                    best_probs, best_ids = probs.topk(top_n, dim=-1)
                    top_p[block_start:block_end] = best_probs.cpu()
                    top_i[block_start:block_end] = best_ids.cpu()

                    true_ids = torch.tensor(seq[block_start:block_end], device=log_probs.device)
                    rows = torch.arange(len(true_ids), device=log_probs.device)
                    actual[block_start:block_end] = probs[rows, true_ids].cpu()  # probability of the real token

                    block_stats = self._token_stats(log_probs, true_ids)
                    for name in TOKEN_STAT_COLUMNS:
                        token_stats[name][block_start:block_end] = block_stats[name].cpu()
                    del logits, log_probs, probs
                done = end
            return top_p, top_i, actual, token_stats

    def _candidates(self, seq, j, top_p, top_i, pos, k, complete_words):
        """Top-k guesses at position j as {lemma: summed probability}, in rank order."""
        # walk down the model's ranked guesses, keeping whole words of the right part of speech;
        # collect up to 3k different lemmas in case the whole-word check removes some
        pool = []                # (token id, lemma, probability)
        lemmas_seen = set()
        for prob, token_id in zip(top_p[j].tolist(), top_i[j].tolist()):
            word = self.vocab_word.get(token_id)
            if word is None:
                continue
            lemma = candidate_lemma(word, pos)
            if not lemma:
                continue
            pool.append((token_id, lemma, prob))
            lemmas_seen.add(lemma)
            if len(lemmas_seen) >= 3 * k:
                break

        if complete_words and pool:
            pool_ids = [token_id for token_id, _lemma, _prob in pool]
            is_whole_word = self._completes_word(seq, j, pool_ids)
            kept = []
            for candidate, keep in zip(pool, is_whole_word):
                if keep:
                    kept.append(candidate)
            pool = kept

        cands = {}
        for _token_id, lemma, prob in pool:
            if lemma in cands:
                cands[lemma] += prob  # 'word' and 'words' are the same guess
            elif len(cands) < k:
                cands[lemma] = prob
        return cands

    def run(self, text, k=5, top_n=300, use_adj=False, max_senses=2, complete_words=True, weighted=False):
        """One row per measurement point: word_idx, char, word, pos, lca_height, lca, candidates,
        cand_mass, actual_word_prob, the token uncertainty values at that word (TOKEN_STAT_COLUMNS)
        (+ lca_weighted if weighted=True).

        Also sets self.last_tokens: one row per token of the text (token_idx, token, char_start,
        char_end and TOKEN_STAT_COLUMNS), i.e. the full per-token time series."""
        encoding = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]  # (char start, char end) of each token
        if self.tok.bos_token_id is not None:
            prefix = [self.tok.bos_token_id]
        else:
            prefix = []
        seq = prefix + ids
        top_p, top_i, actual, token_stats = self._predictions(seq, top_n)
        token_ends = np.array([end for _start, end in offsets])

        token_rows = []
        for t in range(len(ids)):
            j = t + len(prefix)          # position of this token in seq
            token_row = {"token_idx": t, "token": self.tok.decode([ids[t]]),
                         "char_start": offsets[t][0], "char_end": offsets[t][1]}
            for name in TOKEN_STAT_COLUMNS:
                token_row[name] = token_stats[name][j].item()
            token_rows.append(token_row)
        self.last_tokens = pd.DataFrame(token_rows)

        rows = []
        for idx, (char_start, word, pos) in enumerate(content_words(text, use_adj)):
            # index of the first token that ends after the word starts = the token containing the word start
            t = int(np.searchsorted(token_ends, char_start, side="right"))
            j = t + len(prefix)
            if t >= len(ids) or j == 0:
                continue  # no context yet
            text_between = text[offsets[t][0]:char_start]
            if text_between.strip():
                continue  # word doesn't start at a token boundary (e.g. "(dog")

            cands = self._candidates(seq, j, top_p, top_i, pos, k, complete_words)
            if len(cands) < 2:
                continue
            height, ancestor = lca(list(cands), pos, max_senses)
            row = {"word_idx": idx, "char": char_start, "word": word, "pos": pos,
                   "lca_height": height, "lca": name_of(ancestor),
                   "candidates": " ".join(cands), "cand_mass": sum(cands.values()),
                   # each guess's own probability, in the same order as `candidates`
                   "candidate_probs": " ".join(f"{p:.6g}" for p in cands.values()),
                   "actual_word_prob": actual[j].item()}
            for name in TOKEN_STAT_COLUMNS:
                row[name] = token_stats[name][j].item()
            if weighted:
                row["lca_weighted"] = weighted_pair_height(cands, pos, max_senses)
            rows.append(row)
        return pd.DataFrame(rows)

    def free(self):
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# =====================================================================================
# 5. Input texts
# =====================================================================================
N_WORDS = 700
_USER_AGENT = {"User-Agent": "lca-creativity-student-project/0.1"}


def _get(url):
    request = urllib.request.Request(url, headers=_USER_AGENT)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _cut_after_last(text, marker):
    """`text` up to and including the last '.' of `marker` (e.g. '. '); the whole text if marker isn't found."""
    position = text.rfind(marker)
    if position == -1:
        return text
    return text[:position + 1]


def first_words(text, n=N_WORDS):
    """First n words, cut back to the last full sentence."""
    words = text.split()
    cut = " ".join(words[:n])
    return _cut_after_last(cut, ".")


def fetch_wiki(title, n=N_WORDS):
    """Plain text of a Wikipedia article (no headings / references), first n words."""
    query = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                    "format": "json", "titles": title, "redirects": 1})
    response = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{query}"))
    pages = response["query"]["pages"]   # {page id: page}, with exactly one page
    page = list(pages.values())[0]

    body = re.split(r"\n==\s*(See also|References|Notes)\s*==", page["extract"])[0]  # drop the end sections
    body = re.sub(r"\n=+[^=\n]+=+\n", "\n", body)                                   # drop section headings
    one_line = " ".join(body.split())
    return first_words(one_line, n)


def _gutenberg_body(raw):
    """The book text between Project Gutenberg's '*** START OF' and '*** END OF' lines."""
    after_start = raw.split("*** START OF", 1)[1]
    after_start_line = after_start.split("\n", 1)[1]   # skip the rest of the START line
    return after_start_line.split("*** END OF", 1)[0]


def fetch_jokes(fractions=(1 / 3, 2 / 3), n=N_WORDS):
    """Excerpts of *Jokes For All Occasions* (1922, Gutenberg #21084) starting at each fraction of the book."""
    raw = _get("https://www.gutenberg.org/cache/epub/21084/pg21084.txt")
    body = raw.split("*** START OF")[1].split("*** END OF")[0]

    paragraphs = []
    for block in body.split("\r\n\r\n"):
        paragraph = " ".join(block.split())
        is_heading = paragraph.isupper()
        if len(paragraph.split()) > 3 and not is_heading:
            paragraphs.append(paragraph)

    excerpts = []
    for fraction in fractions:
        first = int(len(paragraphs) * fraction)
        chunk = paragraphs[first:first + 40]
        excerpts.append(first_words("\n\n".join(chunk), n))
    return excerpts


def fetch_gutenberg(book_id):
    """(title, body) of a Project Gutenberg book: plain text between the START/END markers."""
    raw = _get(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt")
    title_match = re.search(r"Title:\s*(.+)", raw)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title = str(book_id)
    return title, _gutenberg_body(raw)


def gutenberg_excerpt(body, frac=0.5, n=N_WORDS):
    """~n words starting at the first sentence after `frac` of the book (0.5 = middle), with illustration
    tags, footnote marks and line wrapping removed; ends on a full sentence."""
    text = re.sub(r"\[(Illustration|Footnote)[^\]]*\]", " ", body)   # [Illustration: ...], [Footnote: ...]
    text = re.sub(r"\[\d+\]", "", text)                               # footnote numbers like [1]
    text = re.sub(r"_([^_]+)_", r"\1", text)                          # _italics_ -> italics
    text = " ".join(text.split())                                     # undo line wrapping

    middle = int(len(text) * frac)
    sentence_start = re.compile(r"(?<=[.!?]) (?=[A-Z])")  # a space after . ! ? and before a capital letter
    match = sentence_start.search(text, middle)
    if match:
        start = match.end()
    else:
        start = middle
    words = text[start:].split()[:n]
    return _cut_after_last(" ".join(words), ". ")


def clean_pdf_text(t):
    """Fix the Harry Potter PDF text layer: tabs, U+FFFD quotes, hyphenated line breaks, page numbers."""
    t = t.replace("\t", " ")
    t = re.sub(r"(?<=[A-Za-z])�(?=[A-Za-z])", "'", t)  # I�m -> I'm
    t = t.replace("�", '"')                              # remaining � are quote marks
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)               # de-hyphenate line breaks
    t = re.sub(r"\n\s*\d+\s*\n", "\n", t)                      # bare page numbers
    return re.sub(r"\s+", " ", t).strip()


def hp_excerpt(reader, start, end, n=N_WORDS):
    """~n words from the middle of pages [start, end) of a pypdf.PdfReader."""
    middle_page = (start + end) // 2
    page_texts = []
    for page_number in range(middle_page, middle_page + 6):
        page_text = reader.pages[page_number].extract_text()
        if page_text is None:
            page_text = ""
        page_texts.append(page_text)
    text = clean_pdf_text(" ".join(page_texts))

    first_sentence = re.search(r"(?<=[.!?\"]) (?=[A-Z\"])", text)  # begin at a sentence start
    if first_sentence:
        text = text[first_sentence.end():]
    cut = " ".join(text.split()[:n])

    # end at the last sentence end, either '. ' or '." '
    last_end = max(cut.rfind(". "), cut.rfind('." '))
    if last_end == -1:
        return cut
    return cut[:last_end + 1]


def load_texts(folder):
    """{name: text} for every .txt in `folder` (name = file stem, e.g. 'factual_wiki_volcano')."""
    texts = {}
    for path in sorted(Path(folder).glob("*.txt")):
        texts[path.stem] = path.read_text(encoding="utf-8")
    return texts


# =====================================================================================
# 6. Results + statistics
# =====================================================================================
def kind(text_name):
    """'factual' or 'creative', from the text's file-name prefix."""
    if text_name.startswith("factual"):
        return "factual"
    return "creative"


def slug(model):
    """'Qwen/Qwen2.5-1.5B' -> 'Qwen_Qwen2.5-1.5B' (the CSV file-name prefix)."""
    return model.replace("/", "_")


def load_results(out_dir, models=None):
    """{model_slug: {text: DataFrame}} from the `<model>__<text>.csv` files in out_dir.
    `models` limits it to those models (names or slugs); default = every model found."""
    data = {}
    for path in sorted(Path(out_dir).glob("*__*.csv")):
        model, text = path.stem.split("__", 1)
        if model not in data:
            data[model] = {}
        data[model][text] = pd.read_csv(path)

    if models is not None:
        wanted = set()
        for m in models:
            wanted.add(slug(m))
        selected = {}
        for model, per_text in data.items():
            if model in wanted:
                selected[model] = per_text
        data = selected
    return data


def summarize(df, col="lca_height"):
    """n, mean, var, std, mean |jump| between consecutive points, p90 of one result table."""
    heights = df[col]
    jumps = heights.diff().abs()  # |change| from each point to the next
    return {"n": len(heights), "mean": heights.mean(), "var": heights.var(), "std": heights.std(),
            "mean_abs_jump": jumps.mean(), "p90": heights.quantile(0.9)}


CURVE_FEATURE_NAMES = ["mean", "var", "mean_abs_jump", "max_abs_jump", "share_ge_8", "direction_changes"]


def curve_features(h, high=8):
    """Shape features of one LCA curve (or window): level, spread, jumpiness, share of high points,
    and the rate of direction changes (convergence/divergence). Order = CURVE_FEATURE_NAMES."""
    values = np.asarray(h, dtype=float)
    steps = np.diff(values)          # change from each point to the next
    step_sizes = np.abs(steps)

    level = values.mean()
    spread = values.var()
    average_jump = step_sizes.mean()
    largest_jump = step_sizes.max()
    share_high = (values >= high).mean()

    # direction of each step: +1 up, -1 down, 0 flat; a "change" is a step whose direction
    # differs from the step before it
    directions = np.sign(steps)
    changed = directions[1:] != directions[:-1]
    direction_change_rate = np.mean(changed)

    return [level, spread, average_jump, largest_jump, share_high, direction_change_rate]


def windows(h, win=40, step=20):
    """Slide a window of `win` points with stride `step` over a curve; yields (start, slice)."""
    values = np.asarray(h, dtype=float)
    last_start = len(values) - win
    for start in range(0, last_start + 1, step):
        yield start, values[start:start + win]


def pos_normalize(data, col="lca_height", new_col=None):
    """Z-score `col` within each part of speech, per model, over all its texts.

    Verbs sit ~3 levels lower in WordNet than nouns, so raw heights partly count verbs. After this, each
    point says how high it is *for its POS* (0 = typical noun/verb). Returns a new {model: {text: df}}.
    """
    if new_col is None:
        new_col = col
    out = {}
    for model, per_text in data.items():
        all_points = pd.concat(per_text.values())
        mean_per_pos = all_points.groupby("pos")[col].mean()   # e.g. {'n': 6.45, 'v': 3.06}
        std_per_pos = all_points.groupby("pos")[col].std()

        out[model] = {}
        for text, df in per_text.items():
            own_mean = df["pos"].map(mean_per_pos)   # the average for each row's part of speech
            own_std = df["pos"].map(std_per_pos)
            normalized = df.copy()
            normalized[new_col] = (df[col] - own_mean) / own_std
            out[model][text] = normalized
    return out


def feature_table(data, pos=None, col="lca_height", extra=True):
    """One row per (model, text): model, text, kind + curve_features of its LCA curve.

    data:  output of load_results.
    pos:   None = all points, 'n' = nouns only, 'v' = verbs only.
    extra: also add verb_frac and the model-confidence columns (mean actual_word_prob, mean cand_mass).
    """
    rows = []
    for model, per_text in data.items():
        for text, df in per_text.items():
            if pos is None:
                points = df
            else:
                points = df[df["pos"] == pos]
            row = {"model": model, "text": text, "kind": kind(text), "n": len(points)}
            feature_values = curve_features(points[col])
            for name, value in zip(CURVE_FEATURE_NAMES, feature_values):
                row[name] = value
            if extra:
                row["verb_frac"] = (df["pos"] == "v").mean()
                row["actual_word_prob"] = points["actual_word_prob"].mean()
                row["cand_mass"] = points["cand_mass"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================================
# 7. Embedding spread (an alternative to WordNet LCA)
# =====================================================================================
_EMBEDDING_KEYS = ("wte.weight", "embed_in.weight", "embed_tokens.weight")  # gpt2 / pythia / qwen


def unslug(model_slug):
    """'Qwen_Qwen2.5-1.5B' -> 'Qwen/Qwen2.5-1.5B' (inverse of slug for these model names)."""
    if "_" not in model_slug:
        return model_slug
    owner, name = model_slug.split("_", 1)
    return owner + "/" + name


def _unit_rows(matrix):
    """Scale every row of `matrix` to length 1 (rows of length 0 stay 0)."""
    lengths = matrix.norm(dim=1, keepdim=True)
    return matrix / lengths.clamp(min=1e-12)


@lru_cache(maxsize=4)
def load_token_embeddings(model_name, center=True):
    """(tokenizer, E): the model's input word embeddings, read straight from the cached weights file
    (no full model load, works offline). Rows are unit length; with center=True the mean vector is
    removed first, because LLM embeddings all lean in one shared direction, which makes every pair
    look similar."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from safetensors import safe_open
    from transformers import AutoTokenizer

    try:
        path = hf_hub_download(model_name, "model.safetensors")   # small models: one weights file
    except (EntryNotFoundError, FileNotFoundError, OSError):
        # large models (e.g. Qwen2.5-7B) split the weights over several files; the index says which file
        # holds which tensor, so only the one with the embedding table is opened
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            file_of_tensor = json.load(f)["weight_map"]
        shard = None
        for tensor_name, file_name in file_of_tensor.items():
            if tensor_name.endswith(_EMBEDDING_KEYS):
                shard = file_name
                break
        path = hf_hub_download(model_name, shard)
    with safe_open(path, "pt") as weights:
        embedding_key = None
        for key in weights.keys():
            if key.endswith(_EMBEDDING_KEYS):
                embedding_key = key
                break
        E = weights.get_tensor(embedding_key).float()  # one row per token id
    if center:
        mean_vector = E.mean(dim=0)
        E = E - mean_vector
    E = _unit_rows(E)
    return AutoTokenizer.from_pretrained(model_name), E


def cosine_distance(a, b):
    """1 - cosine similarity of two vectors: 0 = same direction, 1 = unrelated (90 degrees), 2 = opposite."""
    import torch

    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return 1 - similarity.item()


def _pairwise_spread(vectors):
    """(mean, max) cosine distance over every pair of vectors (each pair counted once)."""
    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(cosine_distance(vectors[i], vectors[j]))
    return float(np.mean(distances)), float(np.max(distances))


def word_vector(word, tok, E):
    """Unit vector of ' word' (average of its token vectors if it splits into pieces)."""
    token_ids = tok.encode(" " + word, add_special_tokens=False)
    average = E[token_ids].mean(dim=0)
    return average / average.norm().clamp(min=1e-12)


def embedding_distance(a, b, tok, E):
    """Cosine distance between two words in the model's embeddings."""
    return cosine_distance(word_vector(a, tok, E), word_vector(b, tok, E))


def embedding_spread(words, tok, E):
    """(mean, max) cosine distance over all pairs of `words`: how far apart a set of guesses is in meaning."""
    vectors = []
    for w in words:
        vectors.append(word_vector(w, tok, E))
    return _pairwise_spread(vectors)


def add_embedding_spread(data):
    """Add `emb_mean` and `emb_max` columns to every result table, computed from its saved `candidates`
    with that model's own embeddings. Returns a new {model: {text: df}} (inputs are not modified)."""
    out = {}
    for model, per_text in data.items():
        tok, E = load_token_embeddings(unslug(model))
        out[model] = {}
        for text, df in per_text.items():
            means, maxes = [], []
            for candidates in df["candidates"]:
                mean_distance, max_distance = embedding_spread(candidates.split(), tok, E)
                means.append(mean_distance)
                maxes.append(max_distance)
            with_spread = df.copy()
            with_spread["emb_mean"] = means
            with_spread["emb_max"] = maxes
            out[model][text] = with_spread
    return out


GLOVE_REPO = "sentence-transformers/average_word_embeddings_glove.6B.300d"


@lru_cache(maxsize=1)
def load_glove():
    """(index, E): GloVe 6B 300-d word vectors (400k lowercase words, trained on Wikipedia + news).
    index maps word -> row of E; rows are unit length. First call downloads ~485 MB into the HF cache."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    vocab_path = hf_hub_download(GLOVE_REPO, "0_WordEmbeddings/whitespacetokenizer_config.json")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]       # sorted from most to least common word
    weights_path = hf_hub_download(GLOVE_REPO, "0_WordEmbeddings/model.safetensors")
    with safe_open(weights_path, "pt") as weights:
        E = weights.get_tensor("emb_layer.weight").float()
    E = _unit_rows(E)

    index = {}
    for row, word in enumerate(vocab):
        index[word] = row
    return index, E


def _glove_rows(words):
    """GloVe row numbers of the words GloVe knows (lower-cased), in order."""
    index, _E = load_glove()
    rows = []
    for w in words:
        if w.lower() in index:
            rows.append(index[w.lower()])
    return rows


def glove_distance(a, b):
    """Cosine distance between two words in GloVe (None if a word is missing)."""
    _index, E = load_glove()
    rows = _glove_rows([a, b])
    if len(rows) < 2:
        return None
    return cosine_distance(E[rows[0]], E[rows[1]])


def glove_spread(words):
    """(mean, max) GloVe cosine distance over all pairs of `words`; words GloVe doesn't know are skipped.
    (nan, nan) if fewer than 2 words are known."""
    _index, E = load_glove()
    rows = _glove_rows(words)
    if len(rows) < 2:
        return float("nan"), float("nan")
    vectors = []
    for row in rows:
        vectors.append(E[row])
    return _pairwise_spread(vectors)


def glove_log_rank(words):
    """Average log frequency-rank of `words` in GloVe's word list (which is sorted from most to least common):
    a measure of how rare the words are. Higher = rarer."""
    rows = _glove_rows(words)
    if not rows:
        return float("nan")
    log_ranks = []
    for row in rows:
        log_ranks.append(np.log(row))
    return float(np.mean(log_ranks))


def add_glove_spread(data):
    """Add `glove_mean`, `glove_max` (GloVe spread of each row's saved `candidates`) and `glove_logrank`
    (how rare those guesses are). Returns a new {model: {text: df}}."""
    out = {}
    for model, per_text in data.items():
        out[model] = {}
        for text, df in per_text.items():
            means, maxes, rarity = [], [], []
            for candidates in df["candidates"]:
                words = candidates.split()
                mean_distance, max_distance = glove_spread(words)
                means.append(mean_distance)
                maxes.append(max_distance)
                rarity.append(glove_log_rank(words))
            with_spread = df.copy()
            with_spread["glove_mean"] = means
            with_spread["glove_max"] = maxes
            with_spread["glove_logrank"] = rarity
            out[model][text] = with_spread
    return out


if __name__ == "__main__":
    sanity_check()
