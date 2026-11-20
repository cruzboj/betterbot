"""Hallucinated texts: reading the answer keys, placing a text into the fixed clusters, and comparing a text's
curve word by word with its clean original's.

    planted texts    copies of real texts with {{original => planted | type}} markers; one marker file gives both
                     the clean original and the planted copy, plus where each planted hallucination is
    generated texts  stories Qwen wrote: at LOW_TEMPERATURE, at HIGH_TEMPERATURE, and "switch" stories that are the
                     low-temperature story up to a random switch point and continue at HIGH_TEMPERATURE after it
    placement        the clusters of the 25 real texts stay fixed (docs/adr/0003); a new text goes to the cluster
                     whose centre text's curve is nearest by DTW
    word by word     the planted copy's curve is lined up with its clean original's by DTW, and the gap at each
                     word is kept as it is (nothing averaged)
"""
import re

import numpy as np
import pandas as pd
from tslearn.metrics import dtw, dtw_path

from analysis.cluster import dtw_radius
from qwen_config import GENERATED_DIR, HIGH_TEMPERATURE, LOW_TEMPERATURE, PLANTED_DIR

MARKER = re.compile(r"\{\{(.+?)\s*=>\s*(.+?)\s*\|\s*(entity|relation|invented)\s*\}\}")


# ------------------------------------------------------------------------------------------
# answer keys
# ------------------------------------------------------------------------------------------
def parse_markers(marked_text):
    """(clean original text, planted text, [{original, planted, type, char}, ...]) from one marker file.
    `char` is where the planted word starts in the planted text."""
    clean_parts = []
    planted_parts = []
    edits = []
    planted_length = 0
    position = 0
    for match in MARKER.finditer(marked_text):
        between = marked_text[position:match.start()]
        clean_parts.append(between)
        planted_parts.append(between)
        planted_length += len(between)
        original, planted, kind = match.group(1).strip(), match.group(2).strip(), match.group(3)
        edits.append({"original": original, "planted": planted, "type": kind, "char": planted_length})
        clean_parts.append(original)
        planted_parts.append(planted)
        planted_length += len(planted)
        position = match.end()
    clean_parts.append(marked_text[position:])
    planted_parts.append(marked_text[position:])
    return "".join(clean_parts), "".join(planted_parts), edits


def planted_texts():
    """{planted text name: (clean original's name, planted text, edits)} for every marker file.
    The file is named '<clean original's name>_planted....txt'."""
    texts = {}
    for path in sorted(PLANTED_DIR.glob("*.txt")):
        _clean, planted, edits = parse_markers(path.read_text(encoding="utf-8"))
        original_name = path.stem.split("_planted")[0]
        texts[path.stem] = (original_name, planted, edits)
    return texts


def generated_texts():
    """{generated text name: text} for every story in GENERATED_DIR."""
    texts = {}
    for path in sorted(GENERATED_DIR.glob("*.txt")):
        texts[path.stem] = path.read_text(encoding="utf-8")
    return texts


def switch_points():
    """Table of the switch stories: text, its low-temperature twin, switch token, switch char."""
    path = GENERATED_DIR / "switch_points.csv"
    if not path.exists():
        return pd.DataFrame(columns=["text", "clean original", "switch token", "switch char"])
    return pd.read_csv(path)


def marked_hallucinations():
    """Table of the hallucinations marked by reading the generated stories: text, word, char, reason."""
    path = GENERATED_DIR / "marks.csv"
    if not path.exists():
        return pd.DataFrame(columns=["text", "word", "char", "reason"])
    return pd.read_csv(path)


def row_at_char(word_table, char):
    """The word-table row of the measured word starting at `char`, or None when that word is not measured
    (not a noun/verb, or not found)."""
    rows = np.flatnonzero(word_table["char"].to_numpy() == char)
    if len(rows) == 0:
        return None
    return int(rows[0])


def first_row_from_char(word_table, char):
    """The first measured word at or after `char` (e.g. the first word after a switch point)."""
    rows = np.flatnonzero(word_table["char"].to_numpy() >= char)
    if len(rows) == 0:
        return None
    return int(rows[0])


def curve_position(rows, row):
    """Where table row `row` sits on a curve built from `rows` (see signals.curve_with_rows), or None when that
    word has no value (e.g. GloVe doesn't know it)."""
    if row is None:
        return None
    found = np.flatnonzero(rows == row)
    if len(found) == 0:
        return None
    return int(found[0])


# ------------------------------------------------------------------------------------------
# placing a new text into the fixed clusters (docs/adr/0003)
# ------------------------------------------------------------------------------------------
def place(curve, centre_curves, real_curves):
    """(index of the nearest centre, [DTW distance to each centre]). The DTW window is the one the 25-text
    clustering used, so distances are comparable."""
    radius = dtw_radius(real_curves)
    distances = []
    for centre in centre_curves:
        distances.append(float(dtw(curve, centre, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius)))
    return int(np.argmin(distances)), distances


# ------------------------------------------------------------------------------------------
# word by word
# ------------------------------------------------------------------------------------------
def word_gaps(curve, original_curve):
    """Gap at every point of `curve`: its value minus the clean original's value it is lined up with by DTW
    (no window: the curves are nearly identical apart from the edits). Where DTW lines one point up with several
    original points, the one closest in position is used."""
    path, _distance = dtw_path(curve, original_curve)
    matched = {}
    for i, j in path:
        if i not in matched or abs(j - i) < abs(matched[i] - i):
            matched[i] = j
    gaps = np.zeros(len(curve))
    for i in range(len(curve)):
        gaps[i] = curve[i] - original_curve[matched[i]]
    return gaps


def rank_in_curve(values, position):
    """Rank of the value at `position` among all values of the curve, highest = 1."""
    order = np.argsort(-values, kind="stable")
    return int(np.flatnonzero(order == position)[0]) + 1


def reaction(gaps, position, words_after):
    """At a planted word: its gap, the rank of its |gap| in the whole text, and where the largest |gap| falls
    within the word and the next `words_after` measured words (0 = at the word itself)."""
    window = np.abs(gaps[position:position + words_after + 1])
    return {"gap at word": float(gaps[position]),
            "rank of |gap| in text": rank_in_curve(np.abs(gaps), position),
            "largest |gap| at": f"+{int(np.argmax(window))} words",
            "largest |gap|": float(window.max())}


# ------------------------------------------------------------------------------------------
# the review file (for reading the texts by eye)
# ------------------------------------------------------------------------------------------
def _with_marks_html(text, marks):
    """The text with each marked word in bold red (marks: [(char, word), ...]); HTML, for the side-by-side table."""
    pieces = []
    position = 0
    for char, word in sorted(marks):
        pieces.append(text[position:char])
        pieces.append(f'<b style="color:#c0392b">{text[char:char + len(word)]}</b>')
        position = char + len(word)
    pieces.append(text[position:])
    return "".join(pieces).replace("\n", "<br>")


def temperature_tag(temperature):
    """0.7 -> 't07' (how a generated story's file name says its temperature)."""
    return f"t{round(temperature * 10):02d}"


def write_review(path):
    """texts_hallucinated/REVIEW.md: every hallucinated text in full, with what was planted or marked."""
    lines = ["# Review: hallucinated texts", "",
             "Read the texts and correct the answer keys if needed: planted hallucinations are in the marker files "
             "(`planted/`), hallucinations found in the generated stories are in `generated/marks.csv`.", ""]

    lines += ["## Planted hallucinations", "",
              "~~original~~ → **planted** *(type)*. Only nouns and verbs are measured.", ""]
    for path_in in sorted(PLANTED_DIR.glob("*.txt")):
        marked = path_in.read_text(encoding="utf-8")
        shown = MARKER.sub(lambda match: f"~~{match.group(1).strip()}~~ → **{match.group(2).strip()}** "
                                         f"*({match.group(3)})*", marked)
        original_name = path_in.stem.split("_planted")[0]
        lines += [f"### {path_in.stem}", f"Clean original: `texts/{original_name}.txt`", "", shown, ""]

    stories = generated_texts()
    marks = marked_hallucinations()
    switches = switch_points()
    lines += ["## Generated stories", "",
              f"Left: temperature {LOW_TEMPERATURE}, right: temperature {HIGH_TEMPERATURE} (same request). "
              "Hallucinations found by reading are in red, with the reasons below. The switch story is the "
              f"{LOW_TEMPERATURE} story up to the switch point (‖), then {HIGH_TEMPERATURE}.",
              ""]
    story_keys = []
    for name in stories:
        key = name[len("creative_gen_"):].rsplit("_", 1)[0]
        if key not in story_keys:
            story_keys.append(key)
    for key in story_keys:
        lines += [f"### {key}", ""]
        cells = []
        for suffix in [temperature_tag(LOW_TEMPERATURE), temperature_tag(HIGH_TEMPERATURE)]:
            name = f"creative_gen_{key}_{suffix}"
            if name not in stories:
                cells.append("")
                continue
            text_marks = []
            for _index, mark in marks[marks["text"] == name].iterrows():
                text_marks.append((int(mark["char"]), mark["word"]))
            cells.append(_with_marks_html(stories[name], text_marks))
        lines += [f'<table><tr><th width="50%">temperature {LOW_TEMPERATURE}</th><th>temperature {HIGH_TEMPERATURE}</th></tr>',
                  f'<tr><td valign="top">{cells[0]}</td><td valign="top">{cells[1]}</td></tr></table>', ""]
        switch_name = f"creative_gen_{key}_switch"
        if switch_name in stories:
            switch = switches[switches["text"] == switch_name]
            text = stories[switch_name]
            if len(switch):
                char = int(switch["switch char"].iloc[0])
                text = text[:char] + f" **‖ switch to {HIGH_TEMPERATURE} ‖** " + text[char:]
            lines += [f"**Switch story** (`{switch_name}`)", "", text, ""]
        story_marks = marks[marks["text"].str.startswith(f"creative_gen_{key}_")]
        if len(story_marks):
            lines += ["| text | word | why it is a hallucination |", "|---|---|---|"]
            for _index, mark in story_marks.iterrows():
                lines.append(f"| {mark['text']} | {mark['word']} | {mark['reason']} |")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------
# unusual stretches inside one text (matrix profile, stumpy)
# ------------------------------------------------------------------------------------------
def anomaly_profile(curve, window):
    """Matrix profile of one raw curve: for every stretch of `window` words, the distance (plain Euclidean, raw
    values, nothing rescaled: stumpy.aamp) to the most similar other stretch of the same text.
    High = this stretch is unlike anything else in the text."""
    import stumpy
    return stumpy.aamp(np.asarray(curve, dtype=float), window)[:, 0].astype(float)


def top_anomalies(profile, window, count):
    """Start positions of the `count` most unusual stretches, no two overlapping (most unusual first)."""
    picked = []
    for start in np.argsort(-profile, kind="stable"):
        overlaps = False
        for other in picked:
            if abs(int(start) - other) < window:
                overlaps = True
        if not overlaps:
            picked.append(int(start))
        if len(picked) == count:
            break
    return picked


def stretch_rank(profile, position, window):
    """Rank (1 = most unusual in the text) of the most unusual stretch that contains the word at `position`."""
    first = max(0, position - window + 1)
    last = min(position, len(profile) - 1)
    best = first + int(np.argmax(profile[first:last + 1]))
    return rank_in_curve(profile, best)


def in_top_anomalies(position, starts, window):
    """True when the word at `position` lies inside one of the stretches starting at `starts`."""
    for start in starts:
        if start <= position < start + window:
            return True
    return False
