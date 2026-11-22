from analysis.edits import word_edits

ORIGINAL = "Saturn is the sixth planet from the Sun, and is the second largest planet."


def test_word_edits_finds_each_swapped_word_with_its_place_in_both_texts():
    edited = "Saturn is the sixth planet from the hospital, and is the second largest planet."
    edits = word_edits(ORIGINAL, edited)
    assert edits == [{"char original": ORIGINAL.index("Sun"), "char edited": edited.index("hospital"),
                      "was": "Sun,", "now": "hospital,"}]


def test_word_edits_keeps_later_places_right_after_an_edit_changes_the_length():
    edited = "Saturn is the third planet from the Sun, and is the second largest moon."
    edits = word_edits(ORIGINAL, edited)
    assert [(e["was"], e["now"]) for e in edits] == [("sixth", "third"), ("planet.", "moon.")]
    assert edits[1]["char original"] == ORIGINAL.rindex("planet") and edits[1]["char edited"] == edited.index("moon")


def test_word_edits_joins_neighbouring_changed_words_into_one_edit():
    edited = "Saturn is the sixth planet from my mom, and is the second largest planet."
    edits = word_edits(ORIGINAL, edited)
    assert len(edits) == 1 and edits[0]["was"] == "the Sun," and edits[0]["now"] == "my mom,"


def test_word_edits_is_empty_for_an_unchanged_text():
    assert word_edits(ORIGINAL, ORIGINAL) == []
