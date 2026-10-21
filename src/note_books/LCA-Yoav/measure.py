"""Run an LLM over a text "word by word" and compute the WordNet LCA height of the
model's top-k next-word completions at every content word (noun / verb [/ adj]).

Feeding word by word is done with teacher forcing: one causal forward pass over
the text gives, at the token just before word i, exactly the distribution the
model would produce if fed the prefix text[:i]. The full text (stop words
included) is the model context; stop/filler words only decide *where* we measure
and which completions count (as agreed in the 15.9 meeting).
"""
import re
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordnet_lca import POS, lca

FILLERS = {"also", "being", "however", "thus", "therefore", "would", "could", "should",
           "may", "might", "must", "shall", "said", "say", "says", "get", "got", "make",
           "made", "go", "went", "one", "two", "like", "well", "even", "much", "many", "us"}
STOP = set(stopwords.words("english")) | FILLERS
AUX_LEMMAS = {"be", "have", "do"}  # auxiliary verbs carry no meaning for this test


def base_lemma(word, pos):
    """WordNet lemma, preferring the base form ('hands' -> 'hand', not the
    'hands' = guardianship synset; 'saw' -> 'see')."""
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
    """Lemma of a completion if `pos` is its dominant usage in WordNet (by sense
    frequency), so e.g. 'short'/'back'/'full' don't count as nouns."""
    lemma = base_lemma(word, POS[pos])
    if lemma is None or (pos == "v" and lemma in AUX_LEMMAS):
        return None
    mine = _sense_count(word, POS[pos])
    others = max(_sense_count(word, p) for p in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV) if p != POS[pos])
    return lemma if mine >= others else None


def weighted_pair_height(cands, pos, max_senses=2):
    """Expected LCA height of two *different* completions drawn from the model's
    distribution (renormalised over the candidates): sum_{i<j} p_i p_j h(i,j) / sum_{i<j} p_i p_j.
    A low-probability outlier (e.g. 'vice' at 1%) only enters through its few pairs, instead of
    dragging the whole set to the root as in the max-based lca_height."""
    words, probs = list(cands), list(cands.values())
    num = den = 0.0
    for a in range(len(words)):
        for b in range(a + 1, len(words)):
            w = probs[a] * probs[b]
            num += w * lca([words[a], words[b]], pos, max_senses)[0]
            den += w
    return num / den


def tag_to_pos(tag, use_adj):
    if tag.startswith("NN") and not tag.startswith("NNP"):
        return "n"
    if tag.startswith("VB"):
        return "v"
    if use_adj and tag.startswith("JJ"):
        return "a"
    return None


def content_words(text, use_adj=False):
    """[(char_start, word, pos)] for nouns/verbs (optionally adjectives) that are
    not stop/filler words and exist in WordNet with that POS."""
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


class LCAMeter:
    def __init__(self, model_name, device=None):
        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.window = min(getattr(self.model.config, "max_position_embeddings", 1024) or 1024, 2048)
        self._build_vocab_table()

    def _build_vocab_table(self):
        """token id -> lowercase word, for tokens that start a new alphabetic word.
        (Multi-token words are seen only through their first piece - a known limitation.)"""
        self.vocab_word = {}
        for tid, piece in enumerate(self.tok.convert_ids_to_tokens(list(range(len(self.tok))))):
            if not isinstance(piece, str) or piece[:1] not in ("Ġ", "▁"):
                continue
            w = self.tok.convert_tokens_to_string([piece]).strip()
            if w.isalpha() and w.lower() not in STOP:
                self.vocab_word[tid] = w.lower()
        # tokens that END the current word when they come next: a new word, punctuation, newline, EOS
        n_logits = self.model.get_output_embeddings().weight.shape[0]
        self.boundary = torch.zeros(n_logits, dtype=torch.bool)
        for tid, piece in enumerate(self.tok.convert_ids_to_tokens(list(range(len(self.tok))))):
            if not isinstance(piece, str):
                continue
            s = self.tok.convert_tokens_to_string([piece])
            self.boundary[tid] = piece[:1] in ("Ġ", "▁", "Ċ") or (s[:1] != "" and not s[:1].isalnum())
        if self.tok.eos_token_id is not None:
            self.boundary[self.tok.eos_token_id] = True
        self.boundary = self.boundary.to(self.device)

    @torch.no_grad()
    def _completes_word(self, seq, j, tids, ctx=128):
        """For each candidate token at position j: does the model end the word right after it?
        (P(next token is a word boundary) >= 0.5.) Filters first pieces of longer words such as
        'photo'(synthesis) or 'e', 'sl', 'un' that happen to be WordNet entries."""
        context = seq[max(0, j - ctx):j]
        batch = torch.tensor([context + [t] for t in tids], device=self.device)
        # logits_to_keep=1: only the last position's vocab scores (full logits for a batch of
        # 15 x 129 tokens over Qwen's 152k vocab would be ~600 MB and spill out of an 8 GB GPU)
        probs = torch.softmax(self.model(batch, logits_to_keep=1).logits[:, -1].float(), -1)
        return (probs[:, self.boundary].sum(-1) >= 0.5).tolist()

    @torch.no_grad()
    def _predictions(self, seq, top_n):
        """For every position j>=1 of `seq`: top_n (probs, ids) of P(token_j | seq[:j])
        and the probability of the actual token. Sliding window keeps >= window/2 context."""
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
        """Top-k completions at position j as {lemma: summed probability}, in rank order."""
        pool = []  # (tid, lemma, p) of in-WordNet candidates, more than k in case some get filtered
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
                cands[lemma] += p  # 'word' and 'words' are the same completion
            elif len(cands) < k:
                cands[lemma] = p
        return cands

    def run(self, text, k=5, top_n=300, use_adj=False, max_senses=2, complete_words=True,
            weighted=False):
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
            row = dict(word_idx=idx, char=cstart, word=word, pos=pos, lca_height=h,
                       lca=anc.name() if hasattr(anc, "name") else anc,
                       candidates=" ".join(cands), cand_mass=sum(cands.values()),
                       actual_word_prob=actual[j].item())
            if weighted:  # set aside (didn't change factual vs creative); kept as an option
                row["lca_weighted"] = weighted_pair_height(cands, pos, max_senses)
            rows.append(row)
        return pd.DataFrame(rows)

    def free(self):
        del self.model
        torch.cuda.empty_cache()
