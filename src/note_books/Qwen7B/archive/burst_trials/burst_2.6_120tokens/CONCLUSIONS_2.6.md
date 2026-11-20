# Does one hallucination in a normal text show in how a language model reads it?

## 1. What was checked
Earlier work (notebooks 01–02) found two signals whose curves sort the 25 real texts by kind (factual vs creative):
**GloVe distance of the top 5 guesses** and **LCA height of the likely guesses, at most 3 (10%)**. Each puts 23 of
25 texts in the cluster of their own kind (`results/tables/clusters.csv`).

The question here: take a normal text and make it hallucinate for a few sentences (wrong facts, a change of
subject) by raising the temperature of the model writing it. Does the text's curve move away from the normal text
there, and can the curve, followed word by word, point at the hallucination?

| texts | how many | made by |
|---|---|---|
| clean texts | 3 encyclopedia articles (Paris, Apollo 11, vaccines), 3 stories (magic school, fairy tale, detective) | Qwen2.5-7B-Instruct at temperature 0.7 |
| burst texts | one per clean text | the clean text, one hot stretch, then normal again |

## 2. How the data was collected and calculated

### 2.1 The hot burst
A burst text is its clean original word for word up to a random point (25–75% of its tokens), then 120 tokens (a
few sentences) at temperature 2.6, then back to 0.7 until it is as long as the clean original. All parts keep only
tokens with at least 10% of the top token's probability (`min_p` = 0.1), so even at 2.6 Qwen writes English
sentences, not a jumble.

Shorter or cooler bursts stayed correct: 40 tokens at 2.2 or at 2.6 just continued the text truthfully, because the
correct text before the burst keeps the model on track (`archive/burst_trials`).

What the 120-token bursts contain, read by hand:

| text | burst |
|---|---|
| Paris | clearly wrong: an invented 1926 André Gide novella "Parade", invented artists "Henri Michéline" and "Jean Claude Crach", Place Pigalle "on Rue Alpine" |
| Apollo | slightly wrong: the landing "helped ease Cold War tensions"; instruments left to measure "vibrations due to tidal stresses" |
| Vaccines | correct |
| magic, fairy, detective | the story drifts (a dwarf, "three wishes", a donkey led into a desert; "entered the guest books on the walls") |

### 2.2 Reading
Qwen reads every clean and burst text exactly like the real texts (notebook 01): at each noun and verb it keeps its
top 5 guesses for the next word. The two signals are calculated at every such word, giving each text a curve.

### 2.3 Distance to real writing
At every word, the last 15 words of the curve are compared with the most similar stretch of 15 words anywhere in
the 25 real texts (raw values, Euclidean distance), once against the factual cluster and once against the creative
cluster; the smaller is the **distance to real writing**. Low means "these words read like real writing".

The **normal ceiling** is the highest distance any real text reaches when it is compared without itself
(GloVe 0.617, LCA 12.41).

Because the distance looks back 15 words, a burst can show from its first word until 15 words after it: the
**burst zone**. It covers 16–36% of a text's words (about a quarter).

## 3. Results

### The curves
![GloVe](results/burst/figures/bursts_glove_top5.png)

One row per text: the burst text's raw curve (black) over its clean original's (grey; the same up to the burst),
the burst shaded red, a red dot on each word marked as a wrong fact (`texts_hallucinated/burst/marks.csv`). The
same figure for LCA is `results/burst/figures/bursts_lca_likely_top3_10pct.png`.

Inside the red band the curve swings like everywhere else. Where each marked wrong word ranks among its text's
values (1 = highest):

| text | wrong word | GloVe rank | LCA rank |
|---|---|---|---|
| Paris (197 words) | Parade | 68 | 37 |
| | Ivre | 45 | 20 |
| | Michéline | **1** | 20 |
| | Crach | 170 | 109 |
| | Alpine | 97 | 37 |
| Apollo (170 words) | tidal | 74 | 104 |
| | tensions | 47 | 104 |

### Distance to real writing
Per text and signal (`results/burst/tables/burst_vs_clean.csv`): the highest distance in the burst zone, anywhere
else in the burst text, and anywhere in the clean original.

| text | burst | GloVe: zone / elsewhere / clean | LCA: zone / elsewhere / clean |
|---|---|---|---|
| Paris | clearly wrong | **0.318** / 0.294 / 0.294 | 7.35 / 9.80 / 9.11 |
| Apollo | slightly wrong | 0.253 / 0.292 / 0.334 | 8.00 / 8.37 / 8.31 |
| Vaccines | correct | **0.331** / 0.316 / 0.320 | **7.75** / 7.62 / 7.21 |
| magic | drifts | **0.443** / 0.414 / 0.482 | 7.75 / 7.81 / 8.94 |
| fairy | drifts | **0.386** / 0.378 / 0.378 | **9.17** / 8.19 / 8.19 |
| detective | drifts | 0.383 / 0.395 / 0.365 | 8.00 / 8.89 / 8.06 |

Bold: the text's highest distance is in its burst zone.

## 4. What we can gather
1. **Raising the temperature mostly makes the text wander, not lie.** Of the 6 bursts, only Paris has clearly wrong
   facts and Apollo two doubtful claims; Vaccines stays correct and the stories just take a different path. Even a
   hot model keeps to the facts when the text before it is correct.
2. **The curve does not mark the wrong words.** Of the 7 marked words, 6 sit in the middle of their text's values.
   The exception, the invented name "Michéline", is the highest GloVe value of the whole Paris text, but the other
   invented name next to it ("Crach") is among the lowest; one word is not a pattern.
3. **The distance to real writing notices a change, a little, not a hallucination.** No text goes above what real
   texts reach (at most 0.44 of 0.617, 9.8 of 12.4). The highest distance falls in the burst zone in 6 of 12
   text/signal pairs (about 3 by chance), but the correct Vaccines burst rises as much as the wrong Paris one.
4. **Hallucinating and wandering look the same.** The story bursts behave like the factual ones.
5. **Why:** both signals measure how spread out the model's guesses are. A wrong fact written fluently ("Place
   Pigalle on Rue Alpine") gets guesses as spread as a true one; nothing in these signals checks whether a claim is
   true.

Caveats: 6 texts, one burst each, and only two of the six bursts contain wrong facts (7 words in all); the reference writing is
human, the burst texts are Qwen's.

Next to try: a signal that sees the real word against the model's own expectation at that point (surprisal of
the written word), or comparing each text with clean texts written by the same model rather than with human writing.

## 5. How it was executed in code
| file | what it does |
|---|---|
| `qwen_config.py` | prompts, `COOL_TEMPERATURE`, `BURST_TEMPERATURE`, `BURST_TOKENS`, `BURST_RANGE`, `MIN_P`, `EVOLVING_SIGNALS`, `EVOLVING_WINDOW`, folders |
| `analysis/generate.py` | `write_story` (one text at one temperature), `burst_story` (clean text, hot burst, cool again) |
| `analysis/signals.py` | the signals and each text's curve (shared with notebook 02) |
| `analysis/cluster.py` | DTW + k-medoids into the factual and creative clusters (shared with notebook 02) |
| `analysis/evolving.py` | `distance_to_writing`, `evolving_profile`, `normal_ceiling`, `first_row_from_char` |
| `analysis/figures.py` | `plot_bursts_stacked`: one row per text, raw curves, burst shaded, wrong words dotted |
| `01_run_model.ipynb` | Qwen reads the 25 real texts |
| `02_results.ipynb` | clusters the real texts; picks the two signals |
| `03_hallucination_burst.ipynb` | writes and reads the clean and burst texts, the two stacked figures, the distance table, the answer |
| `tests/` | 14 tests for generate, evolving and figures |

```python
# analysis/generate.py: one hot stretch inside a clean text
hot_ids = write_story(model, tokenizer, prompt, hot_temperature, seed=seed + 2000,
                      story_start_ids=cool_ids[:burst_start], max_new_tokens=burst_tokens, kind=kind, min_p=min_p)
story_ids = write_story(model, tokenizer, prompt, cool_temperature, seed=seed + 3000, story_start_ids=hot_ids,
                        max_new_tokens=max(len(cool_ids) - len(hot_ids), 1), kind=kind, min_p=min_p)
```

The earlier factual-vs-creative write-up and the planted and temperature experiments (the old notebooks 03–06) are in
`archive/hallucination_planted_and_temperature/`.
