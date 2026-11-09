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
