# Only curve classification counts as evidence

A text's kind is judged only by classifying whole curves (comparing curves to each other), never by curve summaries such as the median or SD of a curve. This holds even though the GloVe and Qwen-embedding spread SDs separate the 25 texts perfectly (100% of pairs in order): a single summary number throws away the word-by-word behaviour the research is about, so it cannot support a conclusion about reading a text through the model.

## Consequences

Curve summaries are not shown as results. A future reader seeing a 100% separation in the old notebooks (`archive/`) should not reinstate it as evidence without revisiting this decision.
