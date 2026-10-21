"""Extract one ~700-word excerpt from the middle of each Harry Potter book in
harrypotter.pdf -> texts/creative_hp<N>.txt (local copy only, not redistributed).

The PDF text layer uses tabs for spaces, U+FFFD for curly quotes/apostrophes and
hyphenates across lines; clean() undoes that.
"""
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).parent
N_WORDS = 700
BOOK_STARTS = [11, 281, 572, 948, 1569, 2414, 2975]  # 0-based pages of "CHAPTER ONE"


def clean(t):
    t = t.replace("\t", " ")
    t = re.sub(r"(?<=[A-Za-z])�(?=[A-Za-z])", "'", t)  # I�m -> I'm
    t = t.replace("�", '"')
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)  # de-hyphenate line breaks
    t = re.sub(r"\n\s*\d+\s*\n", "\n", t)  # bare page numbers
    return re.sub(r"\s+", " ", t).strip()


def excerpt(reader, start, end):
    mid = (start + end) // 2
    text = clean(" ".join(reader.pages[p].extract_text() or "" for p in range(mid, mid + 6)))
    first = re.search(r"(?<=[.!?\"]) (?=[A-Z\"])", text)  # begin at a sentence start
    words = text[first.end() if first else 0:].split()
    cut = " ".join(words[:N_WORDS])
    return cut[:max(cut.rfind(". "), cut.rfind('." ')) + 1] or cut


if __name__ == "__main__":
    reader = PdfReader(ROOT / "harrypotter.pdf")
    bounds = BOOK_STARTS + [len(reader.pages)]
    for book, (s, e) in enumerate(zip(bounds, bounds[1:]), 1):
        out = ROOT / "texts" / f"creative_hp{book}.txt"
        out.write_text(excerpt(reader, s, e), encoding="utf-8")
        print(out.name, len(out.read_text(encoding="utf-8").split()), "words")
