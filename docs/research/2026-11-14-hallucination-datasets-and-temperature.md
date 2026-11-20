# Research: hallucination benchmarks, sampling temperature, token-signal detectors

Date: 2026-11-14. Sources are primary only: arXiv abstract/HTML pages, official GitHub repos, Hugging Face dataset cards, and the transformers source. "Verified" means I read the claim on the cited page this session. Anything marked **UNVERIFIED** comes from my prior knowledge of the paper and was not re-read: some PDFs (Holtzman et al., HADES) and the Nature page could not be fetched as text.

## Summary: the facts that most affect the plan

1. **Planted-edit datasets give us exactly what we need: real text plus an answer key.** HADES (BERT word replacement in Wikipedia text, span marked with `===`, binary label) and FAVA training data (Wikipedia passages with LLM-inserted errors, typed by span) both pair an original text with a located error.
2. **Natural-output datasets have span labels but no "clean twin".** RAGTruth (word-level character offsets, about 18k responses) and SelfCheckGPT WikiBio (238 passages, labels per sentence) label errors where models really made them. There is no matching error-free version of the same text, except that WikiBio ships the real Wikipedia paragraph.
3. **HaluEval and TruthfulQA label whole answers or responses.** They do not locate the error inside the text. HaluEval QA is still useful as right/hallucinated answer pairs.
4. **Temperature is literally `logits / T`** (transformers `TemperatureLogitsWarper`, T must be a float > 0). It flattens *every* next-token distribution, so its effect is spread over the whole text. Peer-reviewed evidence links temperature to *incoherence*, and only weakly to novelty (Peeperkorn et al. 2024). Evidence on hallucination rate vs temperature is mixed and mostly small.
5. **Token-probability detectors exist and localize at sentence level.** Examples are SelfCheckGPT's grey-box baselines, HADES's token-level task and LM-Polygraph's token estimators. Sample-consistency methods (SelfCheckGPT, semantic entropy) beat raw token probability, but they need multiple generations.

---

## Q1. Hallucination benchmarks: how errors are made and marked

| Dataset | How errors are produced | Taxonomy | Label unit | Size | License | Clean text + answer key? |
|---|---|---|---|---|---|---|
| **HaluEval** | ChatGPT "sampling-then-filtering" from seed data (HotpotQA, OpenDialKG, CNN/DM). General set: real ChatGPT answers, human-labelled | not detailed in README | whole answer/summary (paired right vs hallucinated); general set Yes/No per response | QA 10k, dialogue 10k, summarization 10k, general 5k | MIT | Partly: QA gives `right_answer` and `hallucinated_answer`, but no error location |
| **TruthfulQA** | Human-written questions "that some humans would answer falsely due to a false belief or misconception" | 38 categories (health, law, finance, politics…) | whole answer | 817 questions | Apache-2.0 (**UNVERIFIED**) | No: it is a question set, not texts with errors |
| **HADES** | Wiki-40B text, words replaced by an off-the-shelf BERT, then multi-round human judgement of whether the span is a hallucination | binary (hallucinated / not) | **one target span per example** (`===span===`, `replaced_ids` token positions) | 8,754 / 1,000 / 1,200 (train/val/test), ~54.5% positive | MIT (repo) | **Yes**, and very close to "swap a word". The repo has the perturbed text. Whether the original is also included is **UNVERIFIED** |
| **FAVA training data** | ~35k Wikipedia articles, rewritten by ChatGPT into genres (blog, tweet…), then ChatGPT/GPT-4 insert tagged errors one type at a time | 6 types: **entity**, **relation** (verb/prep/adj), **sentence** (contradicts evidence), **invented** (non-existent entity), **subjective**, **unverifiable** | **span**, typed, with a correction | 35,074 (30,074 Wiki + 5,000 NQ) | paper CC BY 4.0; data license **UNVERIFIED** | **Yes**: erroneous text, corrected tagged target and the type of each edit |
| **FavaBench** | natural outputs of 3 LMs, human-annotated | same 6 types | span (agreement reported per sentence) | ~1k fine-grained judgements | **UNVERIFIED** | Error location yes, clean twin no |
| **RAGTruth** | natural RAG outputs of GPT-3.5/4, Llama-2-7/13/70B-chat, Mistral-7B-Instruct | Evident/Subtle **Conflict** (contradicts context), Evident/Subtle **Baseless Info** (unsupported addition); flags `implicit_true`, `due_to_null` | **word/char span** (`start`, `end`, `text`, `label_type`) | 2,965 instances, 17,790 responses (QA, summarization, data-to-text) | MIT | Spans yes, no clean twin. Errors are relative to a given context, not world knowledge |
| **SelfCheckGPT WikiBio** | natural GPT-3 biographies of WikiBio people | `accurate` / `minor_inaccurate` / `major_inaccurate` | **sentence** | 238 passages (+20 samples each) | CC-BY-SA-3.0 | Sentence labels, plus the real Wikipedia first paragraph (`wiki_bio_text`) as a factual reference |
| **FActScore** | natural biographies from InstructGPT, ChatGPT, PerplexityAI | supported / not supported | **atomic fact** (score = % supported) | human-annotated bios + 6,500 auto-scored generations from 13 LMs | **UNVERIFIED** | Fact-level labels. Mapping facts back to token positions needs extra work |

Sources:
- HaluEval: https://arxiv.org/abs/2305.11747 (abstract) and https://github.com/RUCAIBox/HaluEval (README, data fields, LICENSE)
- TruthfulQA: https://arxiv.org/abs/2109.07958 (abstract: 817 questions, 38 categories, best model 58% vs humans 94%, larger models less truthful)
- HADES: https://arxiv.org/abs/2104.08704 (abstract) and https://github.com/microsoft/HaDes (README: data creation, sizes, format)
- FAVA: https://arxiv.org/html/2401.06855 (§ taxonomy, § synthetic data creation, § FavaBench)
- RAGTruth: https://arxiv.org/abs/2401.00396 and https://github.com/ParticleMedia/RAGTruth (README)
- WikiBio: https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination (card)
- FActScore: https://arxiv.org/abs/2305.14251 (abstract)

**Usability for our setup:**
- Best fit: **HADES** and **FAVA synthetic data**.
  - Our current pipeline scores a given text under teacher forcing.
  - These datasets give a mostly-factual text with a *known* edited span, which is exactly a "planted edit".
  - FAVA's entity/relation/invented types map to our planned subject swap and wrong-fact cases.
- Natural-error check: **WikiBio** (sentence labels plus a real reference paragraph) and **RAGTruth** (word spans).
  - Caveat for RAGTruth: its errors are only errors relative to a retrieved context, so we would need to score with the context prepended.
- **HaluEval/TruthfulQA** are short QA answers. Our curves need longer texts, so they help little except as answer pairs.

## Q2. What temperature does

**Math (verified):** `TemperatureLogitsWarper.__call__` does `scores_processed = scores / self.temperature`. Softmax is applied afterwards by the sampler, so p_T(v) ∝ exp(z_v / T). `__init__` raises a ValueError unless the temperature is a float strictly greater than 0, and tells you to use `do_sample=False` for greedy decoding. Top-k and top-p are separate warpers, documented as often used together with it. [https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py, class `TemperatureLogitsWarper`]
- Consequences (math, not empirical):
  - T > 1 raises the entropy of *every* step's distribution and gives more mass to the tail.
  - T → 0 approaches argmax.
  - The ranking of tokens is unchanged. Only how sharp the distribution is changes.

**Empirical findings:**
- **Holtzman et al. 2020, "The Curious Case of Neural Text Degeneration"** [https://arxiv.org/abs/1904.09751, abstract, verified]:
  - Maximization (likelihood decoding) gives "bland and strangely repetitive" text.
  - Truncating the "unreliable tail" with nucleus sampling keeps diversity without hurting fluency or coherence.
  - **UNVERIFIED** (PDF unreadable here): §3 and Table 1 report that pure sampling (T=1) is incoherent because of the tail, and that lowering T improves quality at the cost of diversity.
- **Peeperkorn et al. 2024, "Is Temperature the Creativity Parameter of LLMs?"** [https://arxiv.org/abs/2405.00492, abstract, verified]:
  - Setup: fixed model and prompt, narrative generation, only temperature varied.
  - Findings: temperature is weakly correlated with **novelty** and moderately correlated with **incoherence**. It has no relationship with cohesion or typicality.
  - The model used was not named in the abstract.
- **Hallucination rate vs temperature:** no strong primary result found.
  - HalluDial (https://arxiv.org/abs/2406.07070) and a 2026 document-QA study (https://arxiv.org/pdf/2603.08274) were surfaced by search. Search snippets say hallucination rises sharply only at high T, and that T effects are modest compared with model choice.
  - **UNVERIFIED**: I could not confirm these claims on the papers' pages.
  - Renze & Guven (https://arxiv.org/abs/2402.05201) found no significant effect of T in 0-1 on multiple-choice accuracy. This is from a snippet, **UNVERIFIED** in detail.

**Assessment: targeted errors or diffuse incoherence?**
- By construction, temperature changes every step equally (same T at every position).
- So the errors it causes are *not targeted*. A sampled low-probability token can land on a content word, giving a factual error, or on a function word, giving disfluency. After that, teacher-forced continuation conditions on the error and can drift.
- The evidence above (incoherence rises, novelty barely moves) points to **diffuse degradation of the whole text**, not clean, local factual swaps. This is an inference, not a measured result.

## Q3. Token-signal hallucination detectors

- **SelfCheckGPT** [https://arxiv.org/abs/2303.08896, abstract]:
  - Samples N responses and scores each sentence of the main response by its consistency with the samples.
  - It beats grey-box methods in sentence-level AUC-PR on WikiBio.
  - **UNVERIFIED** detail: the grey-box baselines are average and max of −log p and of entropy over a sentence's tokens. These are close to our surprisal and entropy signals.
  - Localization: **sentence**.
- **Semantic entropy** (Farquhar et al., Nature 630, 2024, https://www.nature.com/articles/s41586-024-07421-0 — could not fetch; earlier version Kuhn et al. https://arxiv.org/abs/2302.09664):
  - Samples several answers, clusters them by meaning (bidirectional entailment), and takes the entropy over clusters. The target is "confabulations".
  - **UNVERIFIED**: for long text it splits the text into factoids and scores each one through generated questions, so it localizes at factoid/claim level. It needs sampling plus an NLI model.
- **HADES token-level task** [abstract]: reference-free, per-token detection on planted spans. This is closest to our "local spike" idea.
- **LM-Polygraph token estimators** (see `docs/research/2026-11-07-7b-timeseries-clustering.md` A2): MaximumTokenProbability, TokenEntropy and claim-level aggregates. These are what we already compute.
- **Relevance:** single-pass token probabilities are a known but weaker baseline. Their strength is localization, which our per-word curves already have. Sampling methods are stronger but contradict our "score a given text" design.

---

## Implications for our experiment

- **Use planted edits as the main condition and temperature as a secondary one.**
  - Planted edits (subject swap, entity swap, wrong fact, following the FAVA entity/relation/invented types; or simply reuse HADES/FAVA) give a factual base text, a single known error position, and a clean twin.
  - That allows a paired comparison: curve(edited) − curve(original) at each word.
  - Temperature texts have no answer key and degrade diffusely, so they test "incoherent vs creative", not "hallucinated vs creative".
- **Analysis: local spike detection for edits, whole-curve shape for temperature.**
  - A single swapped entity changes one or a few positions. Whole-text summary features (our SVM features) will dilute it.
  - Use a windowed or peak detector at the known position, scored with AUROC over word positions, like HADES's token-level task.
  - Temperature should shift the whole curve: higher mean surprisal and spread, with more variance.
- **Signals most likely to react:**
  - At the edited word, the **surprisal / P(real word)** of the real token should spike, since the model expected the true entity.
  - **Top-k semantic spread** (LCA height, GloVe spread) *at* the edited word may not change: it describes the model's guesses, which are made before it sees the edit.
  - On the **following** words, expect a delayed reaction. Surprisal and entropy should rise, because the context now conflicts with what the model knows.
  - For temperature texts, expect global increases in surprisal and entropy at every position.
- **Watch for this confound:** creative text is also high in surprisal. The useful contrast is a *local* spike against a stable baseline (edit) versus a uniformly high level (creative or high-T text).
