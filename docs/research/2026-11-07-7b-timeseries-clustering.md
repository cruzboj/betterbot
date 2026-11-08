# Research: Qwen2.5-7B on 8 GB, LM-Polygraph per-token signals, clustering variable-length series

Date: 2026-11-07. Sources are the installed packages in `.venv\Lib\site-packages` (abbreviated `SP\` below), official docs, and PyPI. `SP` = `C:\Users\yoav7\OneDrive\Documentos\GitHub\betterbot\.venv\Lib\site-packages`. Line numbers are approximate. "Checked" means I ran a small CPU-only script in the venv.

## Summary: the facts that most affect the plan

1. **LM-Polygraph 0.7.0 has no teacher-forcing or "score a given text" calculator.** Every white-box path runs `model.generate(...)`, and the token-level stats describe the model's *own* continuation of a prompt. [SP\lm_polygraph\stat_calculators\greedy_probs.py `GreedyProbsCalculator.__call__` ~l.119; infer_causal_lm_calculator.py ~l.186; utils\estimate_uncertainty.py `estimate_uncertainty` ~l.39]
2. **Estimators are plain functions of a `stats` dict** (`Estimator.__call__(stats)`). We can build `greedy_log_probs`, `greedy_log_likelihoods` and `entropy` ourselves from one forward pass over the text and call the estimators directly. [estimator call signatures in each estimator file; utils\causal_lm_with_uncertainty.py ~l.35-50 does exactly this: it builds `deps` and then calls `self.estimator(deps)`]
3. Only **5 estimators are token-level**: `MaximumTokenProbability`, `TokenEntropy`, `PointwiseMutualInformation`, `ConditionalPointwiseMutualInformation` and `SemanticEntropyToken`. Under teacher forcing, `MaximumTokenProbability` becomes the *surprisal of the real token*, −log p(y_t), and PMI is ≈0 when there is no prompt (see A3).
4. Several **sequence-level estimators average a per-token quantity** (SelfCertainty, RenyiNeg, FisherRao, BoostedProb, Perplexity). Their per-token inner terms can be computed as series from the same forward pass.
5. `WhiteboxModel(model, tokenizer, model_path=None, model_type="CausalLM", ...)` accepts an **already-loaded HF model**, so a 4-bit model works. `from_pretrained` is deprecated. [SP\lm_polygraph\utils\model.py `WhiteboxModel.__init__` ~l.399, `from_pretrained` ~l.588]
6. **DBSCAN, silhouette_score and AgglomerativeClustering (except Ward) all accept `metric="precomputed"`.** Ward accepts only Euclidean or l2. **KMeans and KMedoids do not accept a precomputed matrix**, and KMedoids is **not** in scikit-learn 1.9.1 (checked).
7. scikit-learn-extra (which has KMedoids) has **no Python 3.13 wheel** (latest 0.3.0, 2023, newest wheel cp311). For DTW, **tslearn 0.10.0** (pure-Python wheel), **dtaidistance 2.5.1** (cp313 win_amd64 wheel) and **aeon 1.6.0 / sktime 1.2.0** (declare 3.13 and Windows) are all options.
8. **Ward merge height verified:** `Z[i,2] = sqrt(2·n1·n2/(n1+n2))·‖c1−c2‖` (= sqrt(2·ΔSSE)). It matches to 3e-15 on random data, and sklearn's unstructured Ward tree is literally `scipy.cluster.hierarchy.ward(X)`. Same tree and same distances (checked).
9. Qwen2.5-7B-Instruct has **7.61 B parameters**, and its weights alone are about 15.2 GB in bf16. **The current `src/model/load_model.py` (bf16, `device_map="cuda:0"`) cannot fit 8 GB.** Estimated 8-bit size is about 8.7 GB (does not fit). Estimated 4-bit NF4 size is about 5.5-6 GB (fits).
10. In transformers 5.20.2, 4-bit loading uses `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` with `device_map`. `lm_head` is kept unquantized, and logits and hidden states are still returned. **A bnb model cannot be cast with `.to(dtype)`**, which matters for `LCAMeter`, because it calls `.to(self.device)` after loading with a `dtype`.
11. bitsandbytes 0.50.2 **supports Windows 11 with NVIDIA CUDA** (SM75+ recommended; the RTX 4060 is SM89), and the installed wheel ships CUDA 11.8-13.2 DLLs.
12. The venv's torch is CPU-only. **pyproject.toml points uv at `https://download.pytorch.org/whl/cu130`.** pyproject also says `requires-python = "==3.12.*"`, but the venv is Python 3.13.3. That mismatch needs resolving before `uv sync`.

---

## A. LM-Polygraph 0.7.0

### A1. Estimators grouped by what they need
Dependencies are the first argument of each `super().__init__(deps, level)` in `SP\lm_polygraph\estimators\*.py`.

**(a) Token probabilities/logits from one pass, no sampling** (stats `greedy_log_likelihoods`, `greedy_log_probs`, `entropy`, `greedy_lm_log_likelihoods`):
- `MaximumSequenceProbability`, `MaximumTokenProbability` [max_probability.py l.17, l.44]
- `Perplexity` [perplexity.py l.10]
- `MeanTokenEntropy`, `TokenEntropy` [token_entropy.py l.16, l.44]
- `MeanPointwiseMutualInformation`, `PointwiseMutualInformation` [pointwise_mutual_information.py l.16, l.52]. These need `greedy_lm_log_likelihoods`, which is a *second* forward pass over the generation without the prompt [stat_calculators\greedy_lm_probs.py l.27-103].
- `MeanConditionalPointwiseMutualInformation`, `ConditionalPointwiseMutualInformation` [conditional_pointwise_mutual_information.py l.24, l.76]
- `SelfCertainty` [self_certainty.py l.19], `RenyiNeg` [renyi_neg.py l.22], `FisherRao` [fisher_rao.py l.20]
- `BoostedProbSequence` [boostedprob_score.py l.15]. It also reads `greedy_tokens` and calls the external `boostedprob` package (0.1.3, installed).
- `SemanticEntropyToken` [semantic_entropy_token.py l.83]. Its input is `greedy_log_probs`, but it needs a one-off clustering of the whole vocabulary with an external sentence-BERT model (`semantic_bert_path` default `sentence-transformers/bert-base-nli-mean-tokens`), so it is really group (d).
- Prompt/verbal methods that only need generated text: `Linguistic1S`, `Verbalized1S`, `Verbalized2S`, `PTrue`, `PTrueEmpirical`. They need extra generation or prompting, so they are not useful for us.

**(b) Hidden states / attention:**
- `AttentionScore` (`forwardpass_attention_weights`) [attention_score.py l.50]
- `RAUQ` (`attention_all` + `greedy_log_likelihoods`) [rauq.py l.32]
- `Focus` (`attention_all`, `greedy_log_probs`, ...) [focus.py l.322]
- `CSL` (`attention_weights_eliciting_prompt`) [csl.py l.17]
- Embedding-based methods that also need *training-set* embeddings: `MahalanobisDistanceSeq`, `RelativeMahalanobisDistanceSeq`, `RDESeq`, `PPLMDSeq` [mahalanobis_distance.py l.110, relative_mahalanobis_distance.py l.40, rde.py l.70, ppl_md.py l.49]

**(c) Multiple sampled generations:**
- `MonteCarloSequenceEntropy`, `MonteCarloNormalizedSequenceEntropy`, `LexicalSimilarity`, `NumSemSets`, `EigValLaplacian`, `DegMat`, `Eccentricity`, `SemanticEntropy`, `SemanticDensity`, `SentenceSAR`, `SAR`, `EigenScore` (`sample_embeddings`), `KernelLanguageEntropy`, `LUQ`, `PTrueSampling`
- Ensemble estimators `EPT*`/`PET*`/`EPS*`/`PES*`, which need an ensemble of models [ensemble_*_measures.py]

**(d) Extra NLI, embedding or other external model:**
- `ClaimConditionedProbability` (`greedy_tokens_alternatives_nli`) [claim_conditioned_probability.py l.16]
- `TokenSAR` (`token_similarity`) [token_sar.py l.34]
- CoCoA variants (`greedy_sentence_similarity`) [cocoa.py l.12-81]
- Most of group (c), whose `semantic_matrix_*` comes from DeBERTa NLI
- `SemanticEntropyToken` (sentence-BERT)
- All claim-level estimators, which need claim extraction by an external LLM [stat_calculators\extract_claims.py]

### A2. Level (the second argument of `super().__init__`)
- **Token-level:** `MaximumTokenProbability`, `TokenEntropy`, `PointwiseMutualInformation`, `ConditionalPointwiseMutualInformation`, `SemanticEntropyToken`. These are the only ones that natively return a series.
  - Caveat: each drops the last token (`[:-1]`), assuming it is EOS [max_probability.py ~l.60, token_entropy.py ~l.60].
  - Caveat: PMI and CPMI `np.concatenate` all samples in the batch into one flat array [pointwise_mutual_information.py l.78; conditional_... l.109].
  - The ensemble token classes (`EPTtu`, ...) aggregate to `"sequence"` in 0.7.0 [ensemble_token_measures.py l.39].
- **Claim-level:** everything in `estimators\claim\` (`MaximumClaimProbability`, `PerplexityClaim`, `MaxTokenEntropyClaim`, `PointwiseMutualInformationClaim`, `ClaimConditionedProbabilityClaim`, `PTrueClaim`, `AttentionScoreClaim`, `FocusClaim`, `FrequencyScoringClaim`, `TokenSARClaim`, `SelfCertaintyClaim`, `BoostedProbClaim`, `RandomBaselineClaim`) [estimators\claim\*.py].
- **Sequence-level:** all others.

### A3. Scoring a given text (teacher forcing)
- **Not supported out of the box.**
  - `GreedyProbsCalculator` tokenizes the input and calls `model.generate(..., output_scores=True, max_new_tokens=..., min_new_tokens=2)`. `greedy_log_probs` is the stacked generation scores, and `greedy_log_likelihoods[j] = log_probs[j, token_j]` of the *generated* tokens [greedy_probs.py ~l.115-185].
  - `InferCausalLMCalculator` also calls `generate` [infer_causal_lm_calculator.py ~l.186].
  - `estimate_uncertainty` builds a `UEManager` over `Dataset([input_text], [""])` and generates [estimate_uncertainty.py ~l.80-105].
  - A grep for "teacher" or "forced" finds nothing.
  - The only "scoring" helpers are for vLLM/OpenAI-style APIs with pre-extracted logprobs [utils\vllm_with_uncertainty.py, utils\api_with_uncertainty.py]. They are not usable with HF on Windows without vLLM. Whether they could be adapted is UNVERIFIED.
- **Supported workaround:** estimators only read a `stats` dict, so we run one forward pass on the text and build:
  - `greedy_log_probs[0]`: a `(T, V)` array of `log_softmax(logits[t-1])`, the distribution that predicts token t
  - `greedy_log_likelihoods[0]`: `[log p(y_t | y_<t)]` for the real tokens
  - `entropy[0]`: computed with the formula `EntropyCalculator` uses
  - `greedy_tokens[0]`: the real token ids

  Then call `TokenEntropy()(stats)` and so on. `CausalLMWithUncertainty.generate` assembles `deps` and calls `self.estimator(deps)` the same way [utils\causal_lm_with_uncertainty.py ~l.30-50]. Append one dummy position at the end, because token estimators drop the last element.
- **Memory caveat:** a full `(T, V)` float32 array with T≈1000 and V=152,064 is about 0.6 GB per text. It is better to compute the per-token scalars on the GPU in chunks and keep only those.
- Per-token quantities we can compute ourselves from one pass, with the source formula to reproduce:
  - Surprisal of the real token: −log p(y_t | y_<t) [MaximumTokenProbability].
  - Max probability: max_v p(v | y_<t). There is no separate estimator. Under greedy generation the real token *is* the argmax, so this is the faithful "max probability" analogue.
  - Entropy: H_t = −Σ_v p_v log p_v over finite log-probs [stat_calculators\entropy.py `EntropyCalculator.__call__` ~l.46-52].
  - Self-certainty term: −mean_v(log p_v) − log V, i.e. KL(uniform ‖ p) [self_certainty.py ~l.41-54].
  - Rényi term and Fisher-Rao term (formulas in A5).
  - PMI and CPMI, but only with a meaningful "context" x (A5). With no prompt, p(y_t | y_<t, x) = p(y_t | y_<t), so PMI ≡ 0 for t>0. It is only meaningful if, for example, we prepend a prompt such as "Write a factual article:" versus no prompt.

### A4. WhiteboxModel with an already-loaded model
- `WhiteboxModel.__init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_path: str = None, model_type: str = "CausalLM", generation_parameters: GenerationParameters = GenerationParameters(), instruct: bool = False)` stores the objects as-is [utils\model.py ~l.399-421]. So a model loaded with `BitsAndBytesConfig` can be passed in. That this works in practice with 4-bit is UNVERIFIED (not run).
- `WhiteboxModel.from_pretrained(model_path, generation_params={}, add_bos_token=True, **kwargs)` logs "deprecated ... Please instantiate WhiteboxModel directly by passing an already loaded model" and forwards `**kwargs` to `AutoModelForCausalLM.from_pretrained` [~l.588-660].
- `instruct=True` makes `tokenize()` apply the chat template [~l.668-700]. Leave it False for raw-text scoring.
- `WhiteboxModelBasic(model, tokenizer, tokenizer_args, model_type, generation_parameters)` is a lighter adapter [model_adapters\whitebox_model_basic.py l.11].

### A5. One-line definitions of the group (a) estimators
Notation: p = p(· | y_<t, x), y_t = the chosen token, and "higher = more uncertain" throughout.
- MaximumSequenceProbability = −Σ_t log p(y_t) [max_probability.py ~l.33]
- MaximumTokenProbability_t = −log p(y_t) (last token dropped) [~l.60]
- Perplexity = −mean_t log p(y_t). This is the log-perplexity, with no exp [perplexity.py l.15]
- TokenEntropy_t = H_t = −Σ_v p_v log p_v. MeanTokenEntropy = mean_t H_t [token_entropy.py; entropy.py ~l.50]
- PMI_t = −(log p(y_t | y_<t, x) − log p(y_t | y_<t)) for t>0 (for t=0 the second term is 0). MeanPMI = mean over tokens of the same [pointwise_mutual_information.py l.71-78]
- CPMI_t = −(log p(y_t | y_<t, x) − λ·log p(y_t | y_<t)·[t>0 and H_t ≥ τ]), with defaults τ=0.0656, λ=3.599 [conditional_pointwise_mutual_information.py l.16, l.98-109]
- SelfCertainty = −mean_t(−mean_v log p_v − log V) [self_certainty.py ~l.41-54]
- RenyiNeg (α=0.5, T=2): q = softmax(log p / T), s_t = (1/(α−1))·[log Σ_v q_v^α − (α−1)·log V]. Score = mean_t s_t (α=1 branch: Σ q log q + log V) [renyi_neg.py l.20, l.42-62]
- FisherRao (T=2): q as above, s_t = (2/π)·arccos(Σ_v √q_v · √(1/V)). Score = mean_t s_t [fisher_rao.py l.39-54]
- BoostedProbSequence = −mean_t boostedprob_t, where `calculate_boostedprob(log_probs, target)` defaults to `ue_method="sum_dominant_mass"`: the summed probability of "dominant" tokens if y_t is dominant, else p(y_t) [boostedprob_score.py l.34-40; SP\boostedprob ~l.121-146]

## B. Clustering variable-length series (scikit-learn 1.9.1)

### B1. DBSCAN with `metric="precomputed"`
- The matrix X must be square, and the docstring says "a square symmetric matrix" [SP\sklearn\cluster\_dbscan.py `dbscan` docstring ~l.57-61; `DBSCAN` ~l.240].
- A sparse radius-neighbours graph is also allowed, and then only non-zero entries count as neighbours [~l.84, l.151-155].
- `eps`: "The maximum distance between two samples for one to be considered as in the neighborhood of the other... the most important DBSCAN parameter" [~l.63-69].
- `min_samples`: neighbourhood size for a core point, *including the point itself* [~l.71-75].
- Label −1 = noise: "Noisy samples are given the label -1" [~l.132, l.287].
- Choosing eps with the k-distance plot:
  1. Sort every point's distance to its k-th nearest neighbour (k = min_samples − 1, or min_samples if you count the point itself).
  2. Plot the sorted values.
  3. Take eps at the "knee".

  Source: Ester, Kriegel, Sander & Xu, KDD-96, "sorted k-dist graph", §4.2 (cited in the sklearn docstring, ~l.165). Schubert et al. 2017, ACM TODS 42(3):19, revisits this heuristic [docstring ~l.171]. The section details come from the paper as I recall it and were not re-fetched, so treat them as UNVERIFIED.
- With n=25 texts, use a small min_samples (2-4).

### B2. KMeans on variable-length series
- `sklearn.cluster.KMeans` takes feature vectors only. It has no precomputed option. Sound options:
  1. **Resample each series to a fixed length L.** Use `numpy.interp` on a normalised time axis [0,1], or bin into L equal windows and average. Then run KMeans or Ward in Euclidean space. Simple, and Ward-compatible.
  2. **DTW k-means:** `tslearn.clustering.TimeSeriesKMeans(metric="dtw")`. Barycentres are computed with DBA, and the official example fits ragged series built with `to_time_series_dataset`. Euclidean mode "expects a dataset of equal-sized time series". [https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html] tslearn is not installed.
  3. **k-medoids on a precomputed DTW matrix:**
     - `from sklearn.cluster import KMedoids` fails in 1.9.1 (checked; ImportError).
     - KMedoids lives in scikit-learn-extra, whose latest release 0.3.0 (2023-03-27) has wheels only up to cp311 [https://pypi.org/pypi/scikit-learn-extra/json]. So on 3.13 Windows it needs a source build (UNVERIFIED that this builds).
     - Alternatives: aeon or sktime clusterers (they declare 3.13 and Windows, see B3; exact class names UNVERIFIED), or a ~20-line PAM/alternating k-medoids in numpy.

### B3. DTW libraries (Python 3.13 / Windows)
- **tslearn 0.10.0** (2026-10-22): `requires_python >=3.10`, pure-Python `py3-none-any` wheel, so it is installable on 3.13 Windows. Its C/numba dependencies are UNVERIFIED for 3.13. [https://pypi.org/pypi/tslearn/json]
- **dtaidistance 2.5.1** (2026-09-11): `dtaidistance-2.5.1-cp313-cp313-win_amd64.whl` exists [https://pypi.org/pypi/dtaidistance/2.5.1/json]. It provides `dtw.distance_matrix_fast` for distance matrices (function name from the project docs as I recall them; UNVERIFIED here).
- **aeon 1.6.0** (2026-09-18): `>=3.11,<3.15`, classifiers for 3.11-3.14 and Windows [https://pypi.org/pypi/aeon/json].
- **sktime 1.2.0**: `>=3.10,<3.15`, classifiers for 3.10-3.14 and Windows [https://pypi.org/pypi/sktime/json].
- Plain numpy DTW (Sakoe & Chiba, IEEE Trans. ASSP 26(1), 1978):
  - D(0,0)=0, D(i,0)=D(0,j)=∞.
  - D(i,j) = (a_i − b_j)² + min(D(i−1,j), D(i,j−1), D(i−1,j−1)).
  - dist = √D(n,m). An optional Sakoe-Chiba band restricts |i − j·n/m| ≤ w.
  - O(nm) per pair. 25 texts give 300 pairs of about 200×200 cells, which takes seconds in numpy/numba.
  - z-normalise each series first if only the shape matters.

### B4. Other uses of a precomputed matrix
- `silhouette_score(X, labels, metric="precomputed")` is supported. The diagonal must be 0, or it raises "contains non-zero elements on the diagonal. Use np.fill_diagonal(X, 0)." [SP\sklearn\metrics\cluster\_unsupervised.py ~l.83-94, l.250, l.286-297]
- It needs 2 ≤ n_labels ≤ n−1, so exclude or handle DBSCAN noise (−1) yourself. This last point comes from the general sklearn docs as I recall them, so treat it as UNVERIFIED.
- `AgglomerativeClustering(metric="precomputed", linkage in {"complete","average","single"})` is supported [SP\sklearn\cluster\_agglomerative.py docstring ~l.794-799, l.836-841].
- `linkage="ward"` requires `metric in ("euclidean","l2")`, otherwise it raises "Ward can only work with euclidean distances" [~l.1022-1026].
- scipy `linkage` accepts a condensed precomputed matrix (`squareform(D)`). For 'ward', 'centroid' and 'median', "it is the user's responsibility to assure that these distances are in fact Euclidean" [SP\scipy\cluster\hierarchy\_hierarchy_impl.py `linkage` Notes ~l.900].
- DTW is not Euclidean, so do not use Ward on a DTW matrix.

## C. Ward hierarchical clustering

### C1. Criterion, update formula and merge height
- **Criterion:** at each step, merge the pair of clusters whose merge "minimally increases within-cluster variance" [sklearn `ward_tree` docstring, _agglomerative.py ~l.193-196]. scipy calls it "the Ward variance minimization algorithm" [_hierarchy_impl.py ~l.825].
- **Update formula (Lance-Williams form)** used by scipy. Here u = s ∪ t, v is any other cluster, and T = |v|+|s|+|t|:

  d(u,v) = sqrt( (|v|+|s|)/T · d(v,s)² + (|v|+|t|)/T · d(v,t)² − |v|/T · d(s,t)² )

  The starting distances are plain Euclidean distances between points [_hierarchy_impl.py `linkage` docstring ~l.825-840; same text in sklearn `ward_tree` "distances" ~l.248-265].
- The algorithm is nearest-neighbour chain, O(n²) [~l.890; dispatch `_hierarchy.nn_chain` ~l.963]. An observation matrix is first turned into a distance vector with `pdist(y, 'euclidean')` [~l.940].
- **Height:** Z[i,2] = d(s,t) = sqrt(2·|s|·|t|/(|s|+|t|)) · ‖c_s − c_t‖₂ = sqrt(2·ΔSSE), where ΔSSE = (|s||t|/(|s|+|t|))·‖c_s − c_t‖² is the increase in total within-cluster sum of squares.
  - **The proposed formula is correct.** On 25 random 6-D points, the maximum absolute error against `linkage(X,'ward')[:,2]` was 2.7e-15 for both forms (checked).
  - It also follows algebraically from the recurrence: for singletons it gives ‖x_i − x_j‖, and the recurrence preserves it.

### C2. sklearn vs scipy
- With no `connectivity`, `ward_tree` calls `out = hierarchy.ward(X)` and returns `out[:, :2]` as children and `out[:, 2]` as distances [_agglomerative.py ~l.312-319].
- `AgglomerativeClustering(linkage="ward")` uses `ward_tree` (`_TREE_BUILDERS`, ~l.720).
- So the trees are identical. `children_` and `distances_` equal scipy's `Z[:, :2]` and `Z[:, 2]` (checked with `distance_threshold=0`).
- The only difference: sklearn's *structured* Ward (with a connectivity matrix) uses its own heap implementation [~l.330-410].

## D. Qwen/Qwen2.5-7B-Instruct on an 8 GB RTX 4060

### D1. Size and memory
- Model card [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct]:
  - 7.61 B parameters, of which 6.53 B are non-embedding
  - 28 layers, 28 Q heads and 4 KV heads (GQA)
  - Context 131,072 tokens (32,768 in the default config)
  - BF16 safetensors
  - The card gives no GPU memory figure.
- My arithmetic (estimates; UNVERIFIED by measurement):
  - **bf16:** 7.61e9 × 2 B ≈ **15.2 GB** for weights alone.
  - **8-bit (LLM.int8):** transformers says 8-bit "halves the memory-usage" [HF docs below]. Embedding (vocab 152,064 × hidden 3,584 ≈ 0.55 B params) and `lm_head` (another ≈0.55 B, untied in Qwen2.5-7B; the hidden size and untied head are UNVERIFIED from config.json) stay in bf16. So the total is ≈ 6.5 GB + 2.2 GB ≈ **8.7 GB**, which does **not** fit.
  - **4-bit NF4:** "reduces your memory-usage by 4x". 6.53 B × ~0.53 B/param (4 bit + absmax constants; double-quant saves ~0.4 bit/param) ≈ 3.5 GB, plus ≈2.2 GB for bf16 embedding and lm_head, gives ≈ **5.5-6 GB**. That leaves ~2 GB for activations, KV cache and logits.
  - Check the real figure with `model.get_memory_footprint()` [https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes; SP\transformers\modeling_utils.py `get_memory_footprint` ~l.3607].
- **Logits memory:** T≈1000 tokens × 152,064 × 2 B ≈ 0.3 GB in bf16. `.float()` and `softmax` each add about 0.6 GB.
  - `LCAMeter._predictions` does `self.model(window_ids).logits[0]` followed by `softmax` over the whole window (window up to 2048) [lca_core.py ~l.517-521]. This can OOM next to a 5.5 GB model.
  - Compute the logits in chunks: run the base model, then apply `lm_head` to slices of the hidden states. Or reduce each slice immediately to top-k, entropy and the real-token log-prob.

### D2. Loading in 4-bit with transformers 5.20.2
- `BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=False, llm_int8_threshold=6.0, llm_int8_skip_modules=None, llm_int8_enable_fp32_cpu_offload=False, llm_int8_has_fp16_weight=False, bnb_4bit_compute_dtype=None, bnb_4bit_quant_type="fp4", bnb_4bit_use_double_quant=False, bnb_4bit_quant_storage=None, **kwargs)` [SP\transformers\utils\quantization_config.py `BitsAndBytesConfig.__init__` ~l.440-453]. The docs recommend `nf4` and `bnb_4bit_compute_dtype=torch.bfloat16`.
- Recommended call:
  ```python
  bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
              quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda:0")  # or "auto"
  ```
  [HF bitsandbytes docs, sections "Compute data type", "NF4", "Nested quantization"]. transformers 5 uses `dtype=` (not `torch_dtype=`), as `load_model.py` already does.
- `lm_head` stays unquantized: `get_keys_to_not_convert(model)` keeps the output head "in full precision for numerical stability" [SP\transformers\quantizers\base.py ~l.38-41, l.245].
- Logits and hidden states are still produced normally. `Qwen2ForCausalLM.forward(..., logits_to_keep=0)` returns logits for all positions by default (`slice(-logits_to_keep, None)`), and `output_hidden_states` is still accepted [SP\transformers\models\qwen2\modeling_qwen2.py ~l.415-456]. Only the Linear weights are quantized.
- **Casting is blocked.** `model.to(dtype)` on a bnb model raises "You cannot cast a bitsandbytes model in a new `dtype`" [modeling_utils.py `to` ~l.3693-3698]. Moving device with `.to("cuda")` is allowed. Load straight onto the GPU with `device_map`.

### D3. bitsandbytes 0.50 on Windows
- The README table in the installed metadata lists "Windows 11 / Windows Server 2022+, x86-64, NVIDIA GPU cuda: SM60+ minimum, SM75+ recommended" as supported [SP\bitsandbytes-0.50.2.dist-info\METADATA ~l.160-200]. The classifiers include "Operating System :: Microsoft :: Windows" and CUDA 11.8/12/13 [~l.14-21].
- The installed wheel includes `libbitsandbytes_cuda118 … cuda132.dll` [SP\bitsandbytes\].
- The HF docs also say the CUDA backend "is supported on Linux x86-64, Linux aarch64, and Windows platforms", and that NF4/FP4 works on Pascal or newer [HF bitsandbytes docs, "Hardware Compatibility"].
- It needs a CUDA torch build. With the current `torch 2.14.1+cpu`, only the CPU backend loads.

### D4. Where torch should come from
- `pyproject.toml` has `[[tool.uv.index]] url = "https://download.pytorch.org/whl/cu130"` (an unnamed extra index) and `torch>=2.14.0`, `torchvision>=0.29.0`, `torchaudio>=2.11.0` [repo `pyproject.toml`]. So `uv sync` / `uv pip install` is meant to pull CUDA 13.0 builds from that index.
- `requires-python = "==3.12.*"`, but `.venv\pyvenv.cfg` shows Python 3.13.3 [.venv\pyvenv.cfg]. uv will refuse or re-create the env until these agree.
- Whether cu130 Windows cp313 wheels exist for torch 2.14 is UNVERIFIED (not checked).

---

## Implications for our plan
- **Per-token LM-Polygraph signals from one teacher-forced forward pass on a given text.** Compute these ourselves from the logits, using LM-Polygraph's formulas. Alternatively, build the `stats` dict and call the estimator classes directly.
  - Surprisal −log p(y_t) (`MaximumTokenProbability`)
  - Max probability max_v p_v
  - Entropy H_t (`TokenEntropy`)
  - Self-certainty term −mean log p − log V (`SelfCertainty`)
  - Rényi-α and Fisher-Rao terms at T=2 (`RenyiNeg`, `FisherRao`)
  - BoostedProb per token (`boostedprob.calculate_boostedprob`)

  Then sample each series at the measured-word token positions, so it aligns with the LCA series.
- **Not directly usable:**
  - PMI/CPMI, unless we deliberately define a prompt x (two passes per text)
  - SemanticEntropyToken (needs an external sentence-BERT vocabulary clustering)
  - Attention/hidden-state methods (RAUQ and Focus are *sequence*-level; attention for ~1000 tokens × 28 layers × 28 heads is too large on 8 GB)
  - Sampling and NLI methods (they contradict "no generation")
- **Distance + clustering combos for variable-length series (n=25):**
  - DTW matrix (numpy or dtaidistance; z-normalise; optional band) → DBSCAN(precomputed; eps from a sorted k-dist plot; min_samples 2-3), Agglomerative average/complete(precomputed) and silhouette(precomputed). k-medoids would need a custom implementation, aeon/sktime, or tslearn TimeSeriesKMeans(metric="dtw").
  - Resample to fixed length L (for example 100) → KMeans, Ward (scipy or sklearn, same tree), and DBSCAN on Euclidean distance. This is the only route that keeps KMeans and Ward valid.
  - Optional distribution distance that ignores time order (for example a Wasserstein distance between value distributions, `scipy.stats.wasserstein_distance`; not researched here) → precomputed methods only.
- **Loading the 7B model on 8 GB:**
  - Switch `load_model.py` from bf16 to `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` with `dtype=torch.bfloat16` and `device_map="cuda:0"`.
  - Do not call `.to(dtype)` afterwards. `LCAMeter.__init__` currently loads with fp16 and `.to(device)`, so it would need to accept a pre-loaded model.
  - Compute the logits in chunks.
  - First install a CUDA torch from the cu130 index, and fix the 3.12 vs 3.13 `requires-python` mismatch.
