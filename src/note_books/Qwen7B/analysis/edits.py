"""Where a hand-edited text differs from its original (notebook 06): the edited words and their place in both texts."""
import difflib
import re


def _words_with_places(text):
    """([word, ...], [char where it starts, ...]); a word is anything between spaces, punctuation included."""
    words, places = [], []
    for match in re.finditer(r"\S+", text):
        words.append(match.group())
        places.append(match.start())
    return words, places


def word_edits(original, edited):
    """[{char original, char edited, was, now}] for every run of changed words, in reading order.
    Neighbouring changed words form one edit; 'char ...' is where the edit starts in each text."""
    original_words, original_places = _words_with_places(original)
    edited_words, edited_places = _words_with_places(edited)
    matcher = difflib.SequenceMatcher(None, original_words, edited_words, autojunk=False)
    edits = []
    for operation, o_start, o_end, e_start, e_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        # a pure insertion / deletion starts where the next unchanged word is (or at the very end)
        if o_start < len(original_places):
            char_original = original_places[o_start]
        else:
            char_original = len(original)
        if e_start < len(edited_places):
            char_edited = edited_places[e_start]
        else:
            char_edited = len(edited)
        edits.append({"char original": char_original, "char edited": char_edited,
                      "was": " ".join(original_words[o_start:o_end]), "now": " ".join(edited_words[e_start:e_end])})
    return edits
