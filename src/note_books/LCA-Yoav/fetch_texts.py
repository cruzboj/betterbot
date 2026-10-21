"""Download the input texts into texts/.

factual : Wikipedia sections (plain-text extracts via the MediaWiki API)
creative: "Jokes For All Occasions" (1922, original English, public domain, Gutenberg #21084)
Harry Potter is copyrighted and not downloaded: see extract_hp.py (local PDF).
"""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "texts"
N_WORDS = 700
UA = {"User-Agent": "lca-creativity-student-project/0.1"}

WIKI = {"factual_wiki_wordnet": "WordNet",
        "factual_wiki_photosynthesis": "Photosynthesis",
        "factual_wiki_volcano": "Volcano",
        "factual_wiki_mitochondrion": "Mitochondrion",
        "factual_wiki_plate_tectonics": "Plate tectonics",
        "factual_wiki_black_hole": "Black hole",
        "factual_wiki_vaccine": "Vaccine",
        "factual_wiki_copper": "Copper"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8")


def first_words(text, n=N_WORDS):
    words = text.split()
    cut = " ".join(words[:n])
    return cut[:cut.rfind(".") + 1] or cut  # end on a sentence boundary


def wiki(title):
    q = urllib.parse.urlencode(dict(action="query", prop="extracts", explaintext=1, format="json",
                                    titles=title, redirects=1))
    page = next(iter(json.loads(get(f"https://en.wikipedia.org/w/api.php?{q}"))["query"]["pages"].values()))
    body = re.split(r"\n==\s*(See also|References|Notes)\s*==", page["extract"])[0]
    body = re.sub(r"\n=+[^=\n]+=+\n", "\n", body)  # drop section headings
    return first_words(" ".join(body.split()))


def jokes(fractions=(1 / 3, 2 / 3)):
    raw = get("https://www.gutenberg.org/cache/epub/21084/pg21084.txt")
    body = raw.split("*** START OF")[1].split("*** END OF")[0]
    paras = [" ".join(p.split()) for p in body.split("\r\n\r\n") if p.strip()]
    # keep consecutive paragraphs (punchlines are often short); drop only headings
    paras = [p for p in paras if len(p.split()) > 3 and not p.isupper()]
    # excerpts from inside the book, away from the preface
    return [first_words("\n\n".join(paras[int(len(paras) * f):][:40])) for f in fractions]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, title in WIKI.items():
        (OUT / f"{name}.txt").write_text(wiki(title), encoding="utf-8")
        print("saved", name)
    for name, text in zip(["creative_jokes_for_all_occasions", "creative_jokes_2"], jokes()):
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
    print("saved jokes")
