import re
from functools import lru_cache

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize
from nltk.wsd import lesk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download("punkt_tab", quiet=True)

FILLERS = {
    "also", "being", "however", "thus", "therefore", "would", "could", "should",
    "may", "might", "must", "shall", "said", "say", "says", "get", "got", "make",
    "made", "go", "went", "one", "two", "like", "well", "even", "much", "many", "us",
}
# inflected forms of the filler verbs above (a set lookup does not lemmatize)
FILLER_INFLECTIONS = {
    "gets", "getting", "gotten", "goes", "going", "gone",
    "makes", "making", "saying",
}
# extra low-content words: vague nouns/pronouns, hedges, discourse words
EXTRA_FILLERS = {
    "thing", "things", "something", "anything", "nothing", "everything",
    "someone", "anyone", "everyone", "somebody", "anybody", "everybody",
    "way", "lot", "lots", "know", "think", "mean", "seem", "seems", "seemed",
    "really", "still", "yet", "ever", "never", "already", "always", "often",
    "perhaps", "maybe", "rather", "almost", "though", "although", "upon",
    "within", "without", "whether", "another", "every", "oh", "yeah", "ok",
    "okay", "mr", "mrs", "three",
}
AUX_LEMMAS = {"be", "have", "do"}
# contractions as they look after stripping non-letters ("didn't" -> "didnt").
# "hell", "shell", "wed", "shed" are left out on purpose: they are real words.
CONTRACTIONS = {
    "im", "ive", "ill", "id", "theres", "cant", "dont", "wont",
    "arent", "couldnt", "didnt", "doesnt", "hadnt", "hasnt", "havent", "isnt",
    "mightnt", "mustnt", "neednt", "shant", "shouldnt", "wasnt", "werent", "wouldnt",
    "hes", "hed", "shes", "itd", "itll", "thatll", "thats", "whats", "shouldve",
    "theyd", "theyll", "theyre", "theyve", "weve", "youd", "youll", "youre", "youve",
}
custom_stopwords = CONTRACTIONS | {"quite", "thereupon"}


@lru_cache(maxsize=None)
def _adjective_only_lemmas() -> frozenset[str]:
    """WordNet lemmas that are adjectives but never nouns or verbs (slow: scans all of WordNet)."""
    return frozenset(
        set(wn.all_lemma_names(pos=wn.ADJ))
        - set(wn.all_lemma_names(pos=wn.NOUN))
        - set(wn.all_lemma_names(pos=wn.VERB))
    )


@lru_cache(maxsize=None)
def _stopwords(with_adj: bool) -> frozenset[str]:
    words = (set(stopwords.words("english")) | FILLERS | FILLER_INFLECTIONS
             | EXTRA_FILLERS | AUX_LEMMAS | custom_stopwords)
    if with_adj:
        words |= _adjective_only_lemmas()
    return frozenset(words)


def get_stopwords(with_adj: bool = False) -> set[str]:
    """
    Standard English stopwords + fillers (and their inflections) + auxiliaries
    + punctuation-stripped contractions. With `with_adj`, also every WordNet
    lemma that is only ever an adjective. Built once and cached; returns a copy.
    """
    return set(_stopwords(with_adj))


def get_wordnet_pos(nltk_tag: str, use_adj: bool = False) -> str | None:
    """Penn Treebank tag -> WordNet POS ('n' / 'v' / 'a'), or None. Proper nouns are skipped."""
    if nltk_tag.startswith("NN") and not nltk_tag.startswith("NNP"):
        return "n"
    if nltk_tag.startswith("VB"):
        return "v"
    if use_adj and nltk_tag.startswith("JJ"):
        return "a"
    return None

def lesk_target(text_words, clean_target_word):
    """
    lesk - The algorithm reads the words surrounding the target word,
    examines the dictionary definitions of all possible synsets,
    and selects the synset whose definition shares the most words with the context.

    if there is no match will use the first synset
    """
    target_primary_synset = lesk(text_words, clean_target_word)
    
    if not target_primary_synset:
        target_synsets = wn.synsets(clean_target_word)
        if not target_synsets:
            return None 
        target_primary_synset = target_synsets[0]
        
        
    return target_primary_synset

def distance_to_common_lca(synset1, synset2):
    common = synset1.lowest_common_hypernyms(synset2)
    if not common:  # e.g. noun vs verb: no shared ancestor
        return None
    return synset1.shortest_path_distance(common[0])

POS = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ}
VIRTUAL_ROOT = "<ROOT>"

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

def _word_distances(word, pos, max_senses):
    """ancestor -> shortest distance to it from any of the word's allowed senses."""
    shortest = {}
    for sense in senses(word, pos, max_senses):
        for ancestor, distance in _up_distances(sense).items():
            if ancestor not in shortest or distance < shortest[ancestor]:
                shortest[ancestor] = distance
    return shortest

def senses(word, pos, max_senses=2):
    """The word's first `max_senses` senses (WordNet orders them by frequency)."""
    all_senses = wn.synsets(word, pos=pos)
    return all_senses[:max_senses]

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

def _parents(synset):
    """The synsets directly above `synset` in WordNet (normal and 'instance of' hypernyms)."""
    return synset.hypernyms() + synset.instance_hypernyms()