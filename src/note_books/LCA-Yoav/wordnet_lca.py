"""LCA (Least Common Ancestor) height over the WordNet is-a hierarchy.

Score definition
----------------
For a set of words W, the LCA height is the number of is-a (hypernym) levels you
must climb from the farthest word until all words meet at a common ancestor:

    height(W) = min over common ancestors A of  max_{w in W} dist(w, A)

where dist(w, A) is the shortest hypernym path from any allowed sense of w to A.
Small = the words are semantically close (siblings -> 1-2), large = they only
meet near the root (e.g. dog/chair meet at whole.n.02).

Minimising over senses inside dist() (instead of over all sense combinations)
gives the exact answer in O(#ancestors), so it scales to sets of LLM completions.
"""
from functools import lru_cache

from nltk.corpus import wordnet as wn

POS = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ}
VIRTUAL_ROOT = "<ROOT>"  # verbs/adjectives have many roots; join them here


def senses(word, pos, max_senses=2):
    """Allowed senses of `word`: WordNet orders senses by frequency, so the first
    few are the common ones. max_senses=2 keeps e.g. canine.n.02 (the animal; the
    first sense is the tooth) but drops rare senses like dog -> andiron."""
    syns = wn.synsets(word, pos=pos)
    return syns[:max_senses] if max_senses else syns


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
    """Return (height, lca_synset) for a set of words, or (None, None) if any word
    is not in WordNet for this POS."""
    words = list(dict.fromkeys(words))
    dists = [_word_distances(w, POS[pos], max_senses) for w in words]
    if not dists or any(not d for d in dists):
        return None, None
    common = set(dists[0]).intersection(*dists[1:])
    best = min(common, key=lambda a: (max(d[a] for d in dists), a == VIRTUAL_ROOT))
    return max(d[best] for d in dists), best


def lca_height(*words, pos="n", max_senses=2):
    return lca(words, pos, max_senses)[0]


SANITY = [("dog", "cat", 1), ("canine", "cat", 2), ("dog", "chair", 8)]


def sanity_check(max_senses=2):
    print(f"{'pair':<16}{'expected':>9}{'got':>5}  LCA")
    for a, b, exp in SANITY:
        h, anc = lca([a, b], "n", max_senses)
        print(f"{a+'/'+b:<16}{exp:>9}{h:>5}  {anc.name() if hasattr(anc, 'name') else anc}")


if __name__ == "__main__":
    sanity_check()
