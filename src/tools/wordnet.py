import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize
from nltk.wsd import lesk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download("punkt_tab", quiet=True)

def get_stopwords() -> set[str]:
    """
    Returns a set of stopwords that includes the standard English stopwords
    """
    FILLERS = {
        "also", "being", "however", "thus", "therefore", "would", "could", "should",
        "may", "might", "must", "shall", "said", "say", "says", "get", "got", "make",
        "made", "go", "went", "one", "two", "like", "well", "even", "much", "many", "us"
    }
    AUX_LEMMAS = {"be", "have", "do"}
    custom_stopwords = {
        "im", "ive", "ill", "id", "theres", "cant", "dont", "wont", "quite", "thereupon"
    }
    return set(stopwords.words("english")) | FILLERS | AUX_LEMMAS | custom_stopwords

def get_context_blocks(text: str) -> list[str]:
    """
    Splits the input text into context blocks (sentences) using NLTK's sentence tokenizer.
    It also adds some extra abbreviations to the tokenizer to improve sentence splitting.
    """
    tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
    extra_abbreviations = {
        "th", "j", "m", "a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "u.s", "e.g", "i.e", "vs", "etc", "al", "fig", "vol", "no", "st", "dr", "prof"
    }
    tokenizer._params.abbrev_types.update(extra_abbreviations)
    return tokenizer.tokenize(text)

def get_wordnet_pos(nltk_tag: str) -> str | None:
    if nltk_tag.startswith("NN") and not nltk_tag.startswith("NNP"):
        return "n"
    if nltk_tag.startswith("VB"):
        return "v"
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

