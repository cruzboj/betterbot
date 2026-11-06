
Here is the whole pipeline with real numbers from your data. It follows one word, ***think*** in Harry Potter book 3 (gpt2-medium), from the text to the SVM's verdict on that whole text. The example uses the main metric, GloVe with rarity and part of speech removed.

---

## Step 1: The model guesses the word

The text starts:

> "Can I?" "I don't **think** anyone should ride that broom just yet!" said Hermione

*think* is a verb, so it's a measured word. The model reads *"…"I don't"* and predicts the next word. Its top 5 whole-word verb guesses are:

> **know, think, want, believe, understand**

Two numbers come straight from the model (these are the "Confidence" features):
- **P(real word)** = 0.097: it gave *think* about a 10% chance.
- **cand_mass** = 0.506: the 5 guesses together hold about half its probability.

## Step 2: LCA height (WordNet)

Climb WordNet's "is-a" tree from each guess until all 5 meet:

| guess | levels up to the meeting point |
|---|---|
| know | 1 |
| want | 1 |
| understand | 1 |
| think | 3 |
| believe | 3 |

These verbs have no real shared ancestor, so they meet at the virtual root that joins all verb trees. **LCA height = 3**, the climb of the farthest guess.

## Step 3: GloVe spread

Look up each guess's GloVe vector, and compute the cosine distance for all 10 pairs (0 = same meaning, 1 = unrelated):

| pair | distance | pair | distance |
|---|---|---|---|
| know–think | 0.144 | think–believe | 0.226 |
| know–want | 0.196 | think–understand | 0.291 |
| know–understand | 0.204 | want–believe | 0.296 |
| think–want | 0.217 | believe–understand | 0.342 |
| know–believe | 0.256 | want–understand | 0.348 |

**glove_mean = 0.252** (average) and **glove_max = 0.348** (largest). The guesses are close in meaning: they're all "mental" verbs.

## Step 4: Rarity of the guesses

GloVe's word list is sorted from most to least common:

| guess | position in the list | log |
|---|---|---|
| think | 270 | 5.60 |
| want | 304 | 5.72 |
| know | 347 | 5.85 |
| believe | 734 | 6.60 |
| understand | 1,907 | 7.55 |

**glove_logrank = 6.26**, the average. These are common words.

## Step 5: Remove the noun/verb difference

Across all 25 texts, gpt2-medium's **verbs** have an average GloVe spread of **0.627** with SD **0.127**. *think* is a verb, so it's compared with verbs:

$$\text{glove\_z} = \frac{0.252 - 0.627}{0.127} = -2.96$$

The guesses are much tighter than a typical verb's. The same correction for LCA gives (3 − 3.06) / 1.23 = −0.05: completely typical for a verb.

## Step 6: Remove rarity too

For verbs, a straight line fitted over all the points says how spread out guesses usually are at a given rarity:

$$\text{expected} = 0.1197 \times 6.26 - 0.2508 = 0.499$$

$$\text{glove\_r} = \frac{\text{actual} - \text{expected}}{\text{leftover SD}} = \frac{0.252 - 0.499}{0.088} = \mathbf{-2.81}$$

Even for guesses this common, they're unusually close in meaning. This **−2.81** is the final number for this word.

## Step 7: Many words make a curve, and the curve becomes 5 numbers

Steps 1–6 run for every measured word. The first 8 words of *hp3*:

| word | guesses | glove_r |
|---|---|---|
| think | know think want believe understand | −2.81 |
| ride | ask take try know answer | −1.17 |
| broom | bike thing horse bicycle bus | −0.52 |
| looked | look star stand nod laugh | +0.71 |
| think | mean want think know talk | −1.75 |
| going | go think plan get rid | +0.34 |
| sweep | harry look let take ask | +0.09 |
| floor | floor house mud room dust | −0.48 |

Steps between neighbours: +1.65, +0.65, +1.22, −2.46, +2.09, −0.25, −0.56.

From the **whole curve** (177 words for gpt2-medium), compute the 5 shape features. Then do the same for the other two models and average the three:

| | average level | variance | average jump | largest jump | direction changes |
|---|---|---|---|---|---|
| pythia | −0.10 | 0.86 | 1.06 | 3.68 | 0.66 |
| gpt2-medium | −0.08 | 1.03 | 1.13 | 4.38 | 0.68 |
| Qwen | −0.18 | 1.13 | 1.25 | 4.62 | 0.68 |
| **hp3 (average)** | **−0.12** | **1.01** | **1.15** | **4.23** | **0.67** |

*hp3* is now a single point with 5 coordinates.

## Step 8: The SVM labels hp3

**Leave-one-text-out:** the SVM is trained on the other **24 texts**, without *hp3*.

**Standardise.** Each feature is rescaled with the 24 training texts' averages and SDs:

| | level | variance | avg jump | largest jump | dir. changes |
|---|---|---|---|---|---|
| training average | −0.02 | 0.96 | 1.05 | 4.31 | 0.66 |
| training SD | 0.25 | 0.27 | 0.15 | 0.67 | 0.02 |
| **hp3 standardised** | **−0.40** | **+0.18** | **+0.62** | **−0.13** | **+0.48** |

*hp3* sits below average in level and above average in jumpiness. That matches the creative pattern found earlier.

**Predict.** The SVM learned a boundary from the 24 training points; 20 of them (10 per kind) sit near the boundary and define it. *hp3* lands at **−1.35**, clearly on the "creative" side. **Prediction: creative ✓**

## Step 9: The final score

Repeat Step 8 for all 25 texts, each left out in turn:

| | right | wrong |
|---|---|---|
| factual (12) | **7** | copper, mitochondrion, photosynthesis, vaccine, wordnet (all Wikipedia) |
| creative (13) | **12** | hp2 |

$$\text{balanced accuracy} = \frac{7/12 + 12/13}{2} = \frac{0.58 + 0.92}{2} = \mathbf{0.75}$$

That's the **0.75** on the "GloVe, rarity + POS-corrected" bar.

**Every other bar works the same way.** Only Steps 2–6 change: "LCA, raw" uses the height from Step 2 directly, "Word rarity only" uses Step 4's number, "Confidence" uses the two numbers from Step 1, and so on. Steps 7–9 are identical for all of them.
