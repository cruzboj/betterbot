"""Shared LCA toolkit for the betterbot notebooks (lives at the repo root).

# ---- copy this into the first cell of any notebook / script in the repo -------------
# import sys
# from pathlib import Path
# ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "lca_core.py").exists())
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
#     load_texts, fetch_wiki, fetch_jokes, clean_pdf_text, hp_excerpt,
#     # results + stats
#     kind, slug, load_results, summarize, curve_features, CURVE_FEATURE_NAMES,
#     windows, feature_table,
# )
# -------------------------------------------------------------------------------------
# (From the repo root itself, plain `from lca_core import ...` is enough.)

Sections
    1. WordNet LCA          lca, lca_height, weighted_pair_height, sanity_check
    2. Pairwise helpers     get_synset, distance_to_lca, build_lca_graph, draw_lca_graph
    3. Measurement points   base_lemma, candidate_lemma, tag_to_pos, content_words
    4. LLM measurement      LCAMeter (needs torch + transformers, imported on first use)
    5. Input texts          fetch_wiki, fetch_jokes, clean_pdf_text, hp_excerpt, load_texts
    6. Results + stats      kind, slug, load_results, summarize, curve_features, windows, feature_table

Heavy libraries (torch, transformers, networkx, pypdf) are imported only inside the
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

for _res in ["wordnet", "stopwords", "averaged_perceptron_tagger_eng", "punkt_tab"]:
    nltk.download(_res, quiet=True)

from nltk.corpus import stopwords  # noqa: E402  (needs the download above)

__all__ = [
    "POS", "VIRTUAL_ROOT", "FILLERS", "STOP", "AUX_LEMMAS",
    "senses", "lca", "lca_height", "name_of", "weighted_pair_height", "sanity_check",
    "get_synset", "distance_to_lca", "build_lca_graph", "draw_lca_graph",
    "base_lemma", "candidate_lemma", "tag_to_pos", "content_words",
    "LCAMeter",
    "fetch_wiki", "fetch_jokes", "first_words", "clean_pdf_text", "hp_excerpt", "load_texts",
    "kind", "slug", "load_results", "summarize", "curve_features", "CURVE_FEATURE_NAMES", "windows",
    "feature_table",
]


# =====================================================================================
# 1. WordNet LCA
# =====================================================================================
POS = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ}
VIRTUAL_ROOT = "<ROOT>"  # verbs/adjectives have many roots; join them here


def senses(word, pos, max_senses=2):
    """The word's first `max_senses` senses (WordNet orders them by frequency)."""
    return wn.synsets(word, pos=pos)[:max_senses]


@lru_cache(maxsize=200_000)
def _up_distances(synset):
    """Shortest is-a distance from `synset` to each of its ancestors (incl. itself)."""
    dist = {synset: 0}
    frontier = [synset]
    while frontier:
        nxt = []
        for s in frontier:
            for h in s.hypernyms() + s.instance_hypernyms():
                if h not in dist:
                    dist[h] = dist[s] + 1
                    nxt.append(h)
        frontier = nxt
    roots = [s for s in dist if not (s.hypernyms() or s.instance_hypernyms())]
    dist[VIRTUAL_ROOT] = min(dist[r] for r in roots) + 1
    return dist


def _word_distances(word, pos, max_senses):
    """ancestor -> min distance over the word's allowed senses."""
    out = {}
    for s in senses(word, pos, max_senses):
        for a, d in _up_distances(s).items():
            if d < out.get(a, 1 << 30):
                out[a] = d
    return out


def lca(words, pos="n", max_senses=2):
    """(height, lca_synset) for a set of words, or (None, None) if a word isn't in WordNet for this POS.

    height = min over common ancestors A of max_{w in words} dist(w, A): how many is-a levels the
    farthest word climbs until all words meet. Distances are minimised over each word's
    `max_senses` most frequent senses.
    """
    words = list(dict.fromkeys(words))
    dists = [_word_distances(w, POS[pos], max_senses) for w in words]
    if not dists or any(not d for d in dists):
        return None, None
    common = set(dists[0]).intersection(*dists[1:])
    best = min(common, key=lambda a: (max(d[a] for d in dists), a == VIRTUAL_ROOT))
    return max(d[best] for d in dists), best


def lca_height(*words, pos="n", max_senses=2):
    """Just the height: lca_height("dog", "cat") -> 2."""
    return lca(words, pos, max_senses)[0]


def name_of(anc):
    """Printable name of an LCA (synset or the virtual root)."""
    return anc.name() if hasattr(anc, "name") else anc


def weighted_pair_height(cands, pos="n", max_senses=2):
    """Expected LCA height of two different completions drawn from the model's distribution.

    `cands` is {word: probability}. Returns sum_{i<j} p_i p_j h(i,j) / sum_{i<j} p_i p_j, so a
    low-probability outlier only enters through its few pairs instead of pulling the whole set to
    the root as in the max-based `lca`.
    """
    words, probs = list(cands), list(cands.values())
    num = den = 0.0
    for a in range(len(words)):
        for b in range(a + 1, len(words)):
            w = probs[a] * probs[b]
            num += w * lca([words[a], words[b]], pos, max_senses)[0]
            den += w
    return num / den


SANITY = [("dog", "cat", 1), ("canine", "cat", 2), ("dog", "chair", 8)]


def sanity_check(max_senses=2):
    """Print the dog/cat, canine/cat, dog/chair check from the 15.9 meeting."""
    print(f"{'pair':<16}{'estimate':>9}{'got':>5}  LCA")
    for a, b, est in SANITY:
        h, anc = lca([a, b], "n", max_senses)
        print(f"{a + '/' + b:<16}{est:>9}{h:>5}  {name_of(anc)}")


# =====================================================================================
# 2. Pairwise helpers + tree drawing (from LCA.ipynb)
# =====================================================================================
def get_synset(term):
    """'dog' -> first sense of dog; 'canine.n.02' -> that exact synset."""
    return wn.synset(term) if "." in term else wn.synsets(term)[0]


def distance_to_lca(synset1, synset2):
    """NLTK-style distance from synset1 up to its lowest common hypernym with synset2."""
    if isinstance(synset1, str):
        synset1 = get_synset(synset1)
    if isinstance(synset2, str):
        synset2 = get_synset(synset2)
    anc = synset1.lowest_common_hypernyms(synset2)[0]
    return synset1.shortest_path_distance(anc)


def _add_path(graph, synset, anc):
    for path in synset.hypernym_paths():
        if anc in path:
            segment = path[path.index(anc):]
            for parent, child in zip(segment, segment[1:]):
                graph.add_edge(parent.name(), child.name())
            break


def build_lca_graph(pairs, graph=None):
    """networkx DiGraph of the is-a paths from each pair's words up to their LCA.

    pairs: [("dog", "cat"), ("dog", "chair"), ("dog", "canine.n.02"), ...]
    """
    import networkx as nx

    graph = graph if graph is not None else nx.DiGraph()
    for w1, w2 in pairs:
        s1, s2 = get_synset(w1), get_synset(w2)
        anc = s1.lowest_common_hypernyms(s2)[0]
        _add_path(graph, s1, anc)
        _add_path(graph, s2, anc)
    return graph


def draw_lca_graph(graph, out=None, figsize=(14, 8)):
    """Draw a graph from `build_lca_graph` as a left-to-right tree (uses graphviz if installed)."""
    import matplotlib.pyplot as plt
    import networkx as nx

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        graph.graph["rankdir"] = "LR"
        pos = graphviz_layout(graph, prog="dot")
    except Exception:  # no graphviz binary: fall back to layers by depth
        roots = [n for n, d in graph.in_degree() if d == 0]
        depth = {}
        for r in roots:
            for n, d in nx.single_source_shortest_path_length(graph, r).items():
                depth[n] = min(d, depth.get(n, d))
        nx.set_node_attributes(graph, depth, "layer")
        pos = nx.multipartite_layout(graph, subset_key="layer")
    plt.figure(figsize=figsize)
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=300,
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
    forms = wn._morphy(word, pos)
    if not forms:
        return None
    inflected = [f for f in forms if f != word]
    return min(inflected or forms, key=len)


def _sense_count(word, pos):
    lemma = base_lemma(word, pos)
    if lemma is None:
        return -1
    return sum(l.count() for s in wn.synsets(lemma, pos) for l in s.lemmas() if l.name().lower() == lemma)


@lru_cache(maxsize=None)
def candidate_lemma(word, pos):
    """Lemma of a guess if `pos` ('n'/'v'/'a') is its dominant usage in WordNet, else None."""
    lemma = base_lemma(word, POS[pos])
    if lemma is None or (pos == "v" and lemma in AUX_LEMMAS):
        return None
    mine = _sense_count(word, POS[pos])
    others = max(_sense_count(word, p) for p in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV) if p != POS[pos])
    return lemma if mine >= others else None


def tag_to_pos(tag, use_adj=False):
    """Penn Treebank tag -> 'n' / 'v' / 'a' / None (proper nouns are skipped)."""
    if tag.startswith("NN") and not tag.startswith("NNP"):
        return "n"
    if tag.startswith("VB"):
        return "v"
    if use_adj and tag.startswith("JJ"):
        return "a"
    return None


def content_words(text, use_adj=False):
    """[(char_start, word, pos)] of the measurement points in `text`."""
    out, cursor = [], 0
    for sent in nltk.sent_tokenize(text):
        sent_start = text.find(sent, cursor)
        cursor = sent_start + len(sent)
        toks = [(m.start() + sent_start, m.group()) for m in re.finditer(r"[A-Za-z]+", sent)]
        for (start, w), (_, tag) in zip(toks, nltk.pos_tag([w for _, w in toks])):
            pos, lw = tag_to_pos(tag, use_adj), w.lower()
            if pos is None or lw in STOP:
                continue
            lemma = base_lemma(lw, POS[pos])
            if lemma is None or (pos == "v" and lemma in AUX_LEMMAS):
                continue
            out.append((start, w, pos))
    return out


# =====================================================================================
# 4. Measuring a text with an LLM
# =====================================================================================
class LCAMeter:
    """Run a causal LM over a text and score the LCA of its top-k guesses at each content word.

        meter = LCAMeter("gpt2-medium")
        df = meter.run(text, k=5)      # one row per measurement point
        meter.free()
    """

    def __init__(self, model_name, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.window = min(getattr(self.model.config, "max_position_embeddings", 1024) or 1024, 2048)
        self._build_vocab_tables()

    def _build_vocab_tables(self):
        """vocab_word: token id -> lowercase word, for tokens that start a new alphabetic word.
        boundary: tokens that END the current word when they come next (new word, punctuation, newline, EOS)."""
        torch = self.torch
        self.vocab_word = {}
        self.boundary = torch.zeros(self.model.get_output_embeddings().weight.shape[0], dtype=torch.bool)
        for tid, piece in enumerate(self.tok.convert_ids_to_tokens(list(range(len(self.tok))))):
            if not isinstance(piece, str):
                continue
            s = self.tok.convert_tokens_to_string([piece])
            self.boundary[tid] = piece[:1] in ("Ġ", "▁", "Ċ") or (s[:1] != "" and not s[:1].isalnum())
            w = s.strip()
            if piece[:1] in ("Ġ", "▁") and w.isalpha() and w.lower() not in STOP:
                self.vocab_word[tid] = w.lower()
        if self.tok.eos_token_id is not None:
            self.boundary[self.tok.eos_token_id] = True
        self.boundary = self.boundary.to(self.device)

    def _completes_word(self, seq, j, tids, ctx=128):
        """Per candidate token at position j: P(next token is a word boundary) >= 0.5 ?"""
        torch = self.torch
        with torch.no_grad():
            context = seq[max(0, j - ctx):j]
            batch = torch.tensor([context + [t] for t in tids], device=self.device)
            # logits_to_keep=1: only the last position's scores (full logits over Qwen's vocab don't fit 8 GB)
            probs = torch.softmax(self.model(batch, logits_to_keep=1).logits[:, -1].float(), -1)
            return (probs[:, self.boundary].sum(-1) >= 0.5).tolist()

    def _predictions(self, seq, top_n):
        """For every position j>=1: top_n (probs, ids) of P(token_j | seq[:j]) and P(actual token)."""
        torch = self.torch
        with torch.no_grad():
            n, stride = len(seq), self.window // 2
            top_p = torch.full((n, top_n), float("nan"))
            top_i = torch.zeros((n, top_n), dtype=torch.long)
            actual = torch.full((n,), float("nan"))
            done = 1
            while done < n:
                end = min(self.window if done == 1 else done + stride, n)
                start = max(0, end - self.window)
                logits = self.model(torch.tensor([seq[start:end]], device=self.device)).logits[0]
                probs = torch.softmax(logits[done - 1 - start:end - 1 - start].float(), -1)
                p, i = probs.topk(top_n, -1)
                top_p[done:end], top_i[done:end] = p.cpu(), i.cpu()
                tgt = torch.tensor(seq[done:end], device=self.device)
                actual[done:end] = probs.gather(1, tgt[:, None])[:, 0].cpu()
                done = end
            return top_p, top_i, actual

    def _candidates(self, seq, j, top_p, top_i, pos, k, complete_words):
        """Top-k guesses at position j as {lemma: summed probability}, in rank order."""
        pool = []  # (tid, lemma, p); collect up to 3k lemmas in case the whole-word check drops some
        for p, tid in zip(top_p[j].tolist(), top_i[j].tolist()):
            w = self.vocab_word.get(tid)
            lemma = candidate_lemma(w, pos) if w else None
            if lemma:
                pool.append((tid, lemma, p))
                if len({l for _, l, _ in pool}) >= 3 * k:
                    break
        if complete_words and pool:
            ok = self._completes_word(seq, j, [tid for tid, _, _ in pool])
            pool = [c for c, keep in zip(pool, ok) if keep]
        cands = {}
        for _, lemma, p in pool:
            if lemma in cands:
                cands[lemma] += p  # 'word' and 'words' are the same guess
            elif len(cands) < k:
                cands[lemma] = p
        return cands

    def run(self, text, k=5, top_n=300, use_adj=False, max_senses=2, complete_words=True, weighted=False):
        """One row per measurement point: word_idx, char, word, pos, lca_height, lca, candidates,
        cand_mass, actual_word_prob (+ lca_weighted if weighted=True)."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        prefix = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        seq = prefix + ids
        top_p, top_i, actual = self._predictions(seq, top_n)
        ends = np.array([e for _, e in offsets])

        rows = []
        for idx, (cstart, word, pos) in enumerate(content_words(text, use_adj)):
            t = int(np.searchsorted(ends, cstart, side="right"))  # token containing the word start
            j = t + len(prefix)
            if t >= len(ids) or j == 0 or text[offsets[t][0]:cstart].strip():
                continue  # no context yet, or word not at a token boundary (e.g. "(dog")
            cands = self._candidates(seq, j, top_p, top_i, pos, k, complete_words)
            if len(cands) < 2:
                continue
            h, anc = lca(list(cands), pos, max_senses)
            row = dict(word_idx=idx, char=cstart, word=word, pos=pos, lca_height=h, lca=name_of(anc),
                       candidates=" ".join(cands), cand_mass=sum(cands.values()),
                       actual_word_prob=actual[j].item())
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
_UA = {"User-Agent": "lca-creativity-student-project/0.1"}


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as r:
        return r.read().decode("utf-8")


def first_words(text, n=N_WORDS):
    """First n words, cut back to the last full sentence."""
    cut = " ".join(text.split()[:n])
    return cut[:cut.rfind(".") + 1] or cut


def fetch_wiki(title, n=N_WORDS):
    """Plain text of a Wikipedia article (no headings / references), first n words."""
    q = urllib.parse.urlencode(dict(action="query", prop="extracts", explaintext=1, format="json",
                                    titles=title, redirects=1))
    page = next(iter(json.loads(_get(f"https://en.wikipedia.org/w/api.php?{q}"))["query"]["pages"].values()))
    body = re.split(r"\n==\s*(See also|References|Notes)\s*==", page["extract"])[0]
    body = re.sub(r"\n=+[^=\n]+=+\n", "\n", body)
    return first_words(" ".join(body.split()), n)


def fetch_jokes(fractions=(1 / 3, 2 / 3), n=N_WORDS):
    """Excerpts of *Jokes For All Occasions* (1922, Gutenberg #21084) starting at each fraction of the book."""
    raw = _get("https://www.gutenberg.org/cache/epub/21084/pg21084.txt")
    body = raw.split("*** START OF")[1].split("*** END OF")[0]
    paras = [" ".join(p.split()) for p in body.split("\r\n\r\n") if p.strip()]
    paras = [p for p in paras if len(p.split()) > 3 and not p.isupper()]
    return [first_words("\n\n".join(paras[int(len(paras) * f):][:40]), n) for f in fractions]


def clean_pdf_text(t):
    """Fix the Harry Potter PDF text layer: tabs, U+FFFD quotes, hyphenated line breaks, page numbers."""
    t = t.replace("\t", " ")
    t = re.sub(r"(?<=[A-Za-z])�(?=[A-Za-z])", "'", t)
    t = t.replace("�", '"')
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    t = re.sub(r"\n\s*\d+\s*\n", "\n", t)
    return re.sub(r"\s+", " ", t).strip()


def hp_excerpt(reader, start, end, n=N_WORDS):
    """~n words from the middle of pages [start, end) of a pypdf.PdfReader."""
    mid = (start + end) // 2
    text = clean_pdf_text(" ".join(reader.pages[p].extract_text() or "" for p in range(mid, mid + 6)))
    first = re.search(r"(?<=[.!?\"]) (?=[A-Z\"])", text)
    cut = " ".join(text[first.end() if first else 0:].split()[:n])
    return cut[:max(cut.rfind(". "), cut.rfind('." ')) + 1] or cut


def load_texts(folder):
    """{name: text} for every .txt in `folder` (name = file stem, e.g. 'factual_wiki_volcano')."""
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.txt"))}


# =====================================================================================
# 6. Results + statistics
# =====================================================================================
def kind(text_name):
    """'factual' or 'creative', from the text's file-name prefix."""
    return "factual" if text_name.startswith("factual") else "creative"


def slug(model):
    """'Qwen/Qwen2.5-1.5B' -> 'Qwen_Qwen2.5-1.5B' (the CSV file-name prefix)."""
    return model.replace("/", "_")


def load_results(out_dir, models=None):
    """{model_slug: {text: DataFrame}} from the `<model>__<text>.csv` files in out_dir.
    `models` limits it to those models (names or slugs); default = every model found."""
    out_dir = Path(out_dir)
    data = {}
    for f in sorted(out_dir.glob("*__*.csv")):
        model, text = f.stem.split("__", 1)
        data.setdefault(model, {})[text] = pd.read_csv(f)
    if models is not None:
        wanted = {slug(m) for m in models}
        data = {m: d for m, d in data.items() if m in wanted}
    return data


def summarize(df, col="lca_height"):
    """n, mean, var, std, mean |jump| between consecutive points, p90 of one result table."""
    h = df[col]
    return dict(n=len(h), mean=h.mean(), var=h.var(), std=h.std(),
                mean_abs_jump=h.diff().abs().mean(), p90=h.quantile(0.9))


CURVE_FEATURE_NAMES = ["mean", "var", "mean_abs_jump", "max_abs_jump", "share_ge_8", "direction_changes"]


def curve_features(h, high=8):
    """Shape features of one LCA curve (or window): level, spread, jumpiness, share of high points,
    and the rate of direction changes (convergence/divergence). Order = CURVE_FEATURE_NAMES."""
    h = np.asarray(h, dtype=float)
    d = np.diff(h)
    return [h.mean(), h.var(), np.abs(d).mean(), np.abs(d).max(), (h >= high).mean(),
            np.mean(np.sign(d[1:]) != np.sign(d[:-1]))]


def windows(h, win=40, step=20):
    """Slide a window of `win` points with stride `step` over a curve; yields (start, slice)."""
    h = np.asarray(h, dtype=float)
    for s in range(0, len(h) - win + 1, step):
        yield s, h[s:s + win]


def feature_table(data, pos=None, col="lca_height", extra=True):
    """One row per (model, text): model, text, kind + curve_features of its LCA curve.

    data:  output of load_results.
    pos:   None = all points, 'n' = nouns only, 'v' = verbs only.
    extra: also add verb_frac and the model-confidence columns (mean actual_word_prob, mean cand_mass).
    """
    rows = []
    for model, per_text in data.items():
        for text, df in per_text.items():
            sub = df if pos is None else df[df["pos"] == pos]
            row = dict(model=model, text=text, kind=kind(text), n=len(sub))
            row.update(zip(CURVE_FEATURE_NAMES, curve_features(sub[col])))
            if extra:
                row["verb_frac"] = (df["pos"] == "v").mean()
                row["actual_word_prob"] = sub["actual_word_prob"].mean()
                row["cand_mass"] = sub["cand_mass"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sanity_check()
