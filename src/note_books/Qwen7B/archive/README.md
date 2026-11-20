# Archive: earlier analysis (superseded)

These notebooks and figures came before the current pipeline. They are kept for reference with their saved
outputs, and they no longer run: their paths and the `timeseries_tools.py` module were replaced by `config.py` and
`analysis/`.

| file | what it explored | where it went |
|---|---|---|
| `02_timeseries_clustering.ipynb` | first clustering of the curves with 5 signals | `02_results.ipynb` |
| `03_results_explained.ipynb` | factual vs creative value distributions and curve summaries | dropped: curve summaries are not evidence (`docs/adr/0001`) |
| `04_guess_variants.ipynb` | top 1 / top 3 / top 5 / likely-guess versions | now 12 of the 24 signals in `analysis/signals.py` |
| `05_without_alice.ipynb` | the same results without *Alice in Wonderland* | replaced by the memorisation test and the leave-one-out test (`docs/adr/0002`) |

`02_results_robustness.ipynb`, `classify.py`, `robustness.py` and `results/robustness_run/` are the second version:
12 clustering setups judged by a chance score, leave-one-out and DTW-window tests. It was replaced by one simple
method (raw curves, DTW, k-medoids) in `02_results.ipynb`.

`memorisation.py` and `results/memorisation.csv` are the verbatim memorisation test (run once, no text memorised;
see `docs/adr/0002-keep-all-texts.md`).

The first-version results here picked the best of the 12 setups without a chance score or stability tests, so some numbers are
optimistic. Use `CONCLUSIONS.md`.

## Hallucination experiments before the hot burst

`hallucination_planted_and_temperature/` holds the old notebooks 03–06 with their texts, results, the code as it was then
(`code/`) and the earlier `CONCLUSIONS.md`:
- 03–04: hallucinations planted by hand into copies of real texts (e.g. Cyprus → Crete), and Qwen-written stories
  at a higher temperature, checked word by word.
- 05–06: whole articles and stories written cool (0.7), hot (2.6) and cool-then-hot, followed with the evolving
  graph and a normal-ceiling alarm. The hot texts were wrong from the start and ended in word salad; the alarm
  never fired.

They were replaced by `03_hallucination_burst.ipynb`: a normal text with one short hot stretch.
`burst_trials/` holds the bursts that stayed correct (40 tokens at 2.2 and at 2.6) and were not used.
