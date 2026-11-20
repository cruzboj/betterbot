# Hallucinated texts are placed into the fixed 25-text clusters, never reclustered

The two clusters found on the 25 real texts stay fixed. Each new hallucinated text, and each clean generated text, goes into the cluster whose centre text has the nearest raw curve (DTW). The new texts never move the centres and never change which real texts are in the right cluster. This tests what we want to know: does the factual-vs-creative clustering treat a hallucinated text differently from a normal text of its kind?

## Considered options

- **Recluster everything together (25 + new texts, k = 2).** Rejected. The new texts would shift the centres and could change the real-text results, so the experiment would measure how the clustering reacts to new texts, not how hallucinated texts relate to it. The real-text results would also stop being comparable to earlier ones.
- **k = 3, hoping for a "hallucinated" cluster.** Rejected. It treats hallucinated text as a third kind, which the glossary rules out. A hallucination is also one claim inside an otherwise normal text, so a third cluster would more likely collect something else, such as high-temperature style or text length.

Whether a hallucination shows up *where* it happens is a separate, word-level question (curves compared word by word), not settled by cluster placement.
