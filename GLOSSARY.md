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

**Hallucination**:
A claim inside a text that contradicts the text itself (a character renamed, a subject swapped) or the real-world facts it relies on (the capital of France is London). Inventing things is creative; contradicting what the text holds true is a hallucination.
_Avoid_: error, mistake, lie, confabulation

**Hallucinated text**:
A factual or creative text containing at least one hallucination. It keeps its kind; being hallucinated is not a third kind.
_Avoid_: hallucination text, false text, corrupted text

**Clean original**:
A text the model writes at its normal, low temperature; the text a burst copy is compared with.
_Avoid_: source, baseline, reference text, cool text

**Hot burst**:
A few sentences written at a high temperature inside a copy of a clean original, which goes back to normal
temperature afterwards; up to the burst the copy is its clean original word for word.
_Avoid_: switch, hot text, injection

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

**Evolving graph**:
A text's curve drawn word by word as it is read, together with how far the last few words are from real writing of each kind at that moment.
_Avoid_: live graph, streaming plot, running chart

**Distance to real writing**:
At one word, how unlike the last few words of a curve are to the most similar stretch anywhere in real texts of one kind; low means "reads like that kind of writing".
_Avoid_: anomaly score, error, similarity

**Normal ceiling**:
The highest distance to real writing that any real text reaches when compared without itself; anything above it has not been seen in real writing.
_Avoid_: threshold, cutoff, limit

**Burst zone**:
The words from the start of a hot burst until one window of words after it: where the burst can show in the
distance to real writing.
_Avoid_: hallucination region, window

**Reference texts**:
The 25 real (human-written) texts, each labelled with its kind; the only texts a classifier learns from.
Machine-written texts are only labelled, never learnt from.
_Avoid_: training set (when it could include generated text), real data

**Nearest neighbour (1-NN)**:
Labelling a text with the kind of the single reference text whose curve is most similar to its curve.
_Avoid_: KNN (when only one neighbour is meant), matching

**Leave-one-out**:
Scoring a classifier on the reference texts by labelling each one from the other 24, never from itself.
_Avoid_: accuracy (without saying how it was measured), cross-validation

**Spectrum**:
How much of each wave speed a curve holds, from slow (long trends) to fast (word-to-word jitter).
_Avoid_: frequencies, FFT

**Fast share**:
The part of a spectrum's energy that sits in the faster half of the wave speeds; low means the curve moves mostly
in long rises and falls.
_Avoid_: noise level, jitter score

**Classifier**:
Gives a text a kind after learning from the reference texts and their kinds; unlike a classification, it is told
the kinds of the texts it learns from.
_Avoid_: clustering, classification (that is the unlabelled grouping)
