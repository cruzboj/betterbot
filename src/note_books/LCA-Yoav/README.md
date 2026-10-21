# LCA creativity metric

Hypothesis (Eran & Avivit, meeting 15.9): when an LLM reads a **factual** text word by word, its
top-k next-word completions stay semantically close (low WordNet LCA height throughout); in a
**creative** text the completions keep converging and diverging (high variance, jumps).

## Pipeline
**Everything is in [`LCA_Creativity.ipynb`](LCA_Creativity.ipynb)**: one notebook, run top to bottom with the conda
kernel. By default it re-uses the saved model outputs (`RUN_MODELS = False`). The `.py` scripts below are the
original command-line versions and produce the same output.

| step | file |
|---|---|
| LCA over WordNet + sanity check | `wordnet_lca.py` (`python wordnet_lca.py`) |
| input texts | `fetch_texts.py` → `texts/` (Wikipedia sections; Gutenberg #21084 *Jokes For All Occasions*, 1922) |
| preprocessing + LLM measurement | `measure.py` |
| run all models × texts, plots, summary | `run.py` (`--adj` adds adjectives, `--k` completions, `--models ...`); per-text plots of all models + their average in `results/<tag>/per_text/` (`--replot` redraws them from CSVs) |
| bonus SVM on curve shape | `classify.py results/k5` |

Use the conda interpreter: `C:\Users\yoav7\miniconda3\python.exe`.

**Harry Potter** is copyrighted and not downloaded; drop an excerpt (~700 words) in
`texts/creative_harry_potter.txt` and rerun `run.py`. Any `texts/factual_*.txt` /
`texts/creative_*.txt` file is picked up automatically.

## Score definition
`height(W) = min over common ancestors A of max_{w∈W} dist(w, A)`: the number of is-a levels you
climb from the **farthest** word until all words meet (Eran: "the max distance between the LCA and
the lowest word"). Distances use the shortest hypernym path, minimised over each word's 2 most
frequent senses (sense 2 is needed for *canine* = animal, sense 1 is the tooth). Verbs have no
single root in WordNet, so a virtual root joins them.

Sanity check (the 1/2/8 in the meeting were estimates, not exact targets):

| pair | estimate | WordNet | LCA |
|---|---|---|---|
| dog / cat | 1 | 2 | carnivore |
| canine / cat | 2 | 2 | carnivore |
| dog / chair | 8 | 6 | whole (object) |

The ordering is preserved. dog/cat can't be 1 in WordNet 3.0, because `cat.n.01` isn't under
`domestic_animal` (only `house_cat` is), so the two words don't meet at *pet*.

## Measurement details
- "Feeding word by word" = teacher forcing: one causal forward pass gives, at each word, the
  same distribution as feeding the prefix. The model sees the full text; stop/filler words
  (*the, is, also, being*...) and auxiliaries (*be/have/do*) are only skipped as measurement points.
- At each noun/verb (NLTK POS tags), take the top-k completions that are whole words with the **same POS**.
  A completion only counts if that POS is its dominant WordNet usage (so *back/short* don't count as nouns).
  Then compute the set LCA height of those completions.
- Limitation: only single-token completions are seen (most common words are one token).
  Verb heights are systematically lower, because the verb hierarchy is shallow; `pos` is saved
  in the CSVs so the two can be analysed separately.

## Results (2026-10-05, k=5, nouns+verbs, 8 factual / 10 creative texts)
Creative = 7 mid-book Harry Potter excerpts (`extract_hp.py`, from the local PDF), the user's HP
opening excerpt, and 2 unfiltered joke excerpts. Factual = 8 Wikipedia sections. The 3 models agree closely:

| model | kind | mean | var | mean \|jump\| | noun mean | noun var |
|---|---|---|---|---|---|---|
| gpt2-medium | creative | 4.81 | 5.23 | 2.75 | 6.42 | 3.16 |
| | factual | 5.83 | 5.41 | 2.56 | 6.74 | 3.58 |
| pythia-410m | creative | 4.85 | 5.38 | 2.73 | 6.46 | 3.23 |
| | factual | 5.79 | 5.10 | 2.44 | 6.68 | 3.30 |
| Qwen2.5-1.5B | creative | 4.88 | 5.00 | 2.64 | 6.50 | 2.85 |
| | factual | 5.92 | 5.49 | 2.58 | 6.85 | 3.44 |

- The hypothesis is **not supported as stated**: factual text scores *higher* on average, and its
  nouns vary *more*.
- **Confound:** creative texts are 48% verbs vs 26% for factual, and verbs sit ~3 WordNet levels
  lower than nouns. That explains the lower creative mean and most of the slightly higher jumpiness.
- SVM (leave-one-text-out): 87% accuracy on all words, but only 62% (macro-F1 0.61) on **nouns only**.
  The 87% mostly reflects verb density, not LCA dynamics.

## Results v2: `results/k5_clean/` (whole-word completions; probability weighting tried and set aside)
Run with `python run.py` (the default now). `k5/` above is the old run, kept for comparison.
- **Whole-word filter:** a completion only counts if the model would end the word right after it,
  i.e. P(next token = space/punctuation) ≥ 0.5. This cut word-fragment candidates (*e, sl, un, photo*…)
  from ~20% of noun sets to ~2%.
- **`lca_weighted`** (opt-in, `--weighted`): the expected LCA height of two different completions
  drawn by the model's probabilities. **Set aside:** it didn't change the factual vs creative relation.
  The `_w` plots and `per_text_weighted/` in `k5_clean/` are from the one run that had it on.
- **Run time:** the whole-word check is the expensive step, one extra batched model pass per measured
  word (Qwen ~1 min/text instead of ~10 s). `--raw-candidates` skips it.

| metric | words | creative mean | factual mean | SVM accuracy (leave-one-text-out) |
|---|---|---|---|---|
| max (Eran's) | nouns+verbs | 4.78 | 5.67 | 86% |
| max | nouns only | 6.36 | 6.52 | 53% (chance) |
| weighted | nouns+verbs | 3.19 | 3.64 | 77% |
| weighted | nouns only | **4.20** | **4.13** | **66%** |

- Removing fragments did not change the picture for the max metric. On nouns only, the SVM is now at chance level.
- With weighting on nouns only, creative is (slightly) higher than factual for the first time, the
  direction the hypothesis predicts, and that setup gives the best nouns-only classifier (66%).
  The effect is small, so this needs more texts before it's a claim.
- The verb-fraction confound remains in both metrics (creative 47% verbs vs factual 26%).
