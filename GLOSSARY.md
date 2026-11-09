# Factual vs creative text classification

Testing whether a language model, reading a text word by word, can tell if the text is factual or creative.

## Texts

**Kind**:
Whether a text is factual or creative; the label a classification tries to recover.
_Avoid_: class, type, genre, label

**Factual text**:
A text that describes the world as it is: encyclopedia articles and non-fiction books.
_Avoid_: informative, non-fiction, wiki text

**Creative text**:
A text written to tell a story or entertain: fiction and jokes.
_Avoid_: fiction, narrative, imaginative

## Measuring

**Signal**:
One number the model produces at a word while reading a text, e.g. surprisal or the spread of its guesses.
_Avoid_: metric, feature, measurement, estimator

**Curve**:
The sequence of one signal's values over a whole text, in reading order.
_Avoid_: time series, series, trace

**Curve summary**:
A single number describing a whole curve, such as its median or SD. Not accepted as evidence of classification.
_Avoid_: one number per text, statistic

## Results

**Classification**:
Sorting whole texts into two groups by comparing their curves, without being told their kind.
_Avoid_: clustering result, separation

**Cluster**:
One of the two groups of texts formed from how similar their curves are, without knowing their kinds; centred on a real text (its medoid).
_Avoid_: group, class, bucket

**In the right cluster**:
A text whose cluster is made up mostly of texts of its own kind.
_Avoid_: correct, accuracy, sorted correctly

**Text map**:
A 2D picture where each text is a dot and close dots have similar curves.
_Avoid_: embedding plot, scatter, projection
