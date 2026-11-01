"""Add non-Wikipedia factual and non-Harry-Potter creative texts to texts/.

~700 words from the middle of each public-domain Project Gutenberg book. Both groups come from the same
site and a similar era (1850-1930), so factual vs creative is no longer the same as Wikipedia vs Harry Potter.

    python fetch_gutenberg.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "lca_core.py").exists())
sys.path.insert(0, str(ROOT))
from lca_core import fetch_gutenberg, gutenberg_excerpt  # noqa: E402

OUT = Path(__file__).parent / "texts"

BOOKS = {
    "factual_gut_relativity": 30155,       # Einstein, Relativity: The Special and General Theory
    "factual_gut_candle": 14474,           # Faraday, The Chemical History of a Candle
    "factual_gut_origin_species": 1228,    # Darwin, On the Origin of Species
    "factual_gut_outline_science": 20417,  # Thomson, The Outline of Science
    "creative_gut_alice": 11,              # Carroll, Alice's Adventures in Wonderland
    "creative_gut_sherlock": 1661,         # Doyle, The Adventures of Sherlock Holmes
    "creative_gut_oz": 55,                 # Baum, The Wonderful Wizard of Oz
    "creative_gut_grimm": 2591,            # Grimm, Grimms' Fairy Tales
}

if __name__ == "__main__":
    for name, book_id in BOOKS.items():
        title, body = fetch_gutenberg(book_id)
        text = gutenberg_excerpt(body)
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"{name:32s} #{book_id:<6} {len(text.split()):4d} words  {title}")
