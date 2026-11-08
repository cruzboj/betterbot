# Keep all 25 texts, including Alice in Wonderland

All 25 texts are used, including *Alice in Wonderland*, even though the confidence signals tend to put it with the factual texts: the model's #1 guess is the real word 74% of the time for *Alice*, against 28–49% for the other creative texts.

## Considered options

Excluding memorised texts was considered. A one-off verbatim test (the model continues 5 passages per text by itself) found no memorised text. *Alice* matched 15% and so did the Wikipedia article on copper; every other text matched 2–11%. *Alice* is predictable (famous, simple prose), not memorised. Removing it would only remove a text that is hard for some signals, so it stays.

The test was then removed from the pipeline as not needed for the results. Its code and scores are in `src/note_books/Qwen7B/archive/`. `EXCLUDED_TEXTS` in `qwen_config.py` can still leave a text out by name.
