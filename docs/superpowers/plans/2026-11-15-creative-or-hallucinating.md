# Creative or Hallucinating: the Evolving Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Have Qwen write factual articles and creative stories, and raise its temperature mid-text so it hallucinates (topic changes, wrong facts like "the capital of France is London"). Then use the two best signals' curves, drawn as they grow word by word, to tell creative writing apart from hallucinating.

**Architecture:**
- **Texts.** New texts are generated in notebook 05, read by Qwen exactly as in notebook 01, and kept apart from the 25 real texts.
- **The evolving graph.** At every word, the last `EVOLVING_WINDOW` words of a text's raw curve are compared with the most similar stretch anywhere in the real texts of each fixed cluster (matrix-profile AB-join, raw values). This gives two lines over reading time: "distance to factual writing" and "distance to creative writing".
- **The idea being tested:**
  - Creative writing stays close to the creative texts.
  - Hallucinating drifts away from *both* kinds of real writing, or a factual article turns creative after the switch.
  - An **alarm** is raised at the first word where the distance to the nearest real writing goes above the highest value any real text ever reaches.

**Tech Stack:** Python (miniconda: `C:/Users/yoav7/miniconda3/python.exe`), transformers (`generate` with `min_p`), stumpy (`aamp` AB-join), tslearn (existing clustering), matplotlib, plotly, pytest.

**Spec:** There is no separate spec. This plan implements the user's request of 2026-11-15:
- Use what was found, i.e. GloVe distance of the top 5 guesses (`glove_top5`) and LCA of the likely guesses, at most 3, at 10% (`lca_likely_top3_10pct`).
- Make texts with hallucinations (sudden topic changes mid-sentence, wrong facts) by raising the generating model's temperature.
- Tell creativity apart from hallucination with the graph that evolves over time.
- The supervisor's brief: *"לפגישה הבאה תנסו לקחת כמה טקסטים עם הזיות … משנים פתאום באמצע המשפט את הנושא מוסיפים מידע לא נכון כמו בירה צרפת היא לונדון … על ידי העלאת הטמפרטורה של המודל הגנרטיבי ותנסו לראות אם אפשר להבחין בין יצירתיות להזיות להשתמש בגרף המתפתח בזמן"*.
- Standing decisions: `GLOSSARY.md`, `docs/adr/0003-hallucinated-texts-join-fixed-clusters.md` (new texts never join the clustering), and `docs/research/2026-11-14-hallucination-datasets-and-temperature.md`.

## Global Constraints

- **Raw curves only.** No averages, medians, SDs, smoothing or standardising. DTW and matrix-profile distances on raw values are fine. The alarm threshold is a maximum, not an average.
- **The fixed clusters stay fixed** (ADR 0003). Only the 25 real texts form them; new texts are compared against them.
- **Signals:** `EVOLVING_SIGNALS = ["glove_top5", "lca_likely_top3_10pct"]`.
- **Code layout:**
  - Logic goes in `src/note_books/Qwen7B/analysis/`, settings in `qwen_config.py`. Notebooks are the source of truth for runs.
  - Explicit loops, readable names, one-line docstrings. Inside notebook cells use `#` comments, never docstrings.
- **Results stay separate:**
  - Texts: `texts_hallucinated/temperature/`
  - Model run: `results/run_temperature/{words,tokens}`
  - Outputs: `results/temperature/{figures,tables}`
- **Don't break existing work:**
  - Notebooks 03 and 04 and their outputs stay as they are.
  - New function parameters default to the old behaviour.
  - `lca_core.py` (shared with a partner) is not changed.
- **Never commit.** The user commits on request. No step below commits.
- **Commands:**
  - Run notebooks with `JUPYTER_PATH=<scratchpad>/jp`. That kernel spec points to the miniconda Python.
  - Run tests from `src/note_books/Qwen7B` with `C:/Users/yoav7/miniconda3/python.exe -m pytest tests -v`.

## Review Focus

1. **A planted text compared against its own clean original** gives near-zero distance everywhere except at the edit. That is not a fair test, so the clean original must be left out of the references. Pinned by `test_profile_leaves_out_named_texts` (Task 3).
2. **A real text measured against a reference set that contains itself** gives distance 0, which makes the ceiling meaningless. Pinned by `test_ceiling_leaves_each_text_out` (Task 3).
3. **A hot text that degenerates** (non-English, few measured nouns or verbs) can be shorter than the window. The profile must come back all NaN without raising. Pinned by `test_short_curve_is_all_nan` (Task 3).
4. **A reference curve shorter than the window** must be skipped, not crash stumpy. Pinned by `test_short_reference_is_skipped` (Task 3).
5. **The switch point can fall after the last measured word, or the profile can be all NaN.** `first_alarm` returns `None`, and the alarm table shows "no alarm", not a crash. Pinned by `test_first_alarm_none_when_never_above` and `test_first_alarm_ignores_nan` (Task 3).

---

### Task 1: Factual articles, min_p, and picking the hot temperature

**Why:**
- Wrong facts need factual texts, so stories alone are not enough.
- With the full vocabulary, T ≥ 1.15 broke Qwen into mixed scripts. `min_p` cuts only the tokens below a share of the top token's probability, so a higher temperature can make Qwen wrong without making it garbage.
- The cool and hot halves use the same `min_p`, so temperature is the only thing that changes at the switch.

**Files:**
- Modify: `src/note_books/Qwen7B/analysis/generate.py`
- Modify: `src/note_books/Qwen7B/qwen_config.py`
- Create: `src/note_books/Qwen7B/tests/conftest.py`, `src/note_books/Qwen7B/tests/test_generate.py`

**Interfaces:**
- Produces:
  - `request_text(kind: str, prompt: str) -> str`, where kind is `"creative"` or `"factual"`.
  - `story_request(tokenizer, prompt, device, kind="creative")`
  - `write_story(model, tokenizer, prompt, temperature, seed, story_start_ids=None, max_new_tokens=GENERATION_MAX_TOKENS, kind="creative", min_p=0.0) -> list[int]`
  - `switch_story(model, tokenizer, prompt, low_story_ids, low_temperature_seed, high_temperature, rng, kind="creative", min_p=0.0)`
- Produces, in config:
  - `FACT_PROMPTS = {"paris": "Paris, the capital of France", "apollo": "the Apollo 11 Moon landing", "vaccines": "how vaccines train the immune system"}`
  - `MIN_P = 0.1`
  - `COOL_TEMPERATURE = 0.7`
  - `HOT_TEMPERATURE = 1.5` (the default; Step 6 sets the final value)
  - `HOT_TEMPERATURES_TO_PROBE = [1.3, 1.5, 1.8]`
  - Folders: `TEMPERATURE_DIR = HALLUCINATED_DIR / "temperature"`, `TEMPERATURE_RUN_DIR = HERE / "results" / "run_temperature"` (with `TEMPERATURE_WORDS_DIR` and `TEMPERATURE_TOKENS_DIR`), `TEMPERATURE_FIGURES_DIR` and `TEMPERATURE_TABLES_DIR` under `HERE / "results" / "temperature"`
  - Analysis: `EVOLVING_SIGNALS = ["glove_top5", "lca_likely_top3_10pct"]`, `EVOLVING_WINDOW = 15` (measured words, about 5–10% of a text)

- [ ] **Step 1: Write `tests/conftest.py`** so that it puts `src/note_books/Qwen7B` at the front of `sys.path`, then write the failing tests:

```python
from analysis.generate import request_text

def test_creative_request_is_a_story():
    text = request_text("creative", "a lost wand")
    assert "story" in text and "a lost wand" in text and "650" in text

def test_factual_request_is_an_article_of_facts():
    text = request_text("factual", "Paris, the capital of France")
    assert "encyclopedia" in text and "Paris, the capital of France" in text and "650" in text
    assert "story" not in text

def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        request_text("poem", "x")
```

- [ ] **Step 2: Run** `python -m pytest tests/test_generate.py -v`. Expected: FAIL with an ImportError on `request_text`.

- [ ] **Step 3: Implement `request_text`.**
  - The factual wording: `"Write an encyclopedia article of about {STORY_WORDS} words about {prompt}. Write only the article itself: no title, no headings, no notes."`
  - The creative wording stays the current one.
  - `story_request` gets `kind` and builds its message from `request_text`. `write_story` and `switch_story` pass `kind` and `min_p` through, and `generate()` gets `min_p=min_p` (0.0 = off, which is the old behaviour).
  - Add the config values listed above. Update the `generate.py` docstring: temperature plus `min_p`, with the reason.

- [ ] **Step 4: Run** the tests. Expected: 3 passed.

- [ ] **Step 5: Create `05_temperature_texts.ipynb`.** Its sections 0–2 are setup, the planted table and loading the model, the same as notebook 03 cells 1–2. Section 3 is "Pick the hot temperature":
  - Write 250 tokens for `FACT_PROMPTS["paris"]` at each value in `HOT_TEMPERATURES_TO_PROBE`, with `MIN_P` and seed `RANDOM_SEED`.
  - Print each text in full.
  - Save the texts to `TEMPERATURE_DIR / "probe.txt"`.

- [ ] **Step 6: Run section 3 and read the three texts.** The hot temperature is the highest one whose text is still English sentences and has at least one wrong fact or topic change. Set it as `HOT_TEMPERATURE` in config and put the reason in the comment. If none qualifies, try `MIN_P = 0.05` or `0.2` and ask the user before going on.

### Task 2: Write and read the temperature texts (notebook 05)

**Files:**
- Modify: `src/note_books/Qwen7B/05_temperature_texts.ipynb` (sections 4–5)
- Create: `texts_hallucinated/temperature/*.txt`, `switch_points.csv`, `marks.csv`

**Interfaces:**
- Consumes: Task 1's `write_story`, `switch_story`, `FACT_PROMPTS`, `STORY_PROMPTS`, `MIN_P`, `COOL_TEMPERATURE` and `HOT_TEMPERATURE`, and the existing `LCAMeter.run`.
- Produces:
  - 18 texts named `{kind}_temp_{key}_{cool|switch|hot}`, where kind is `factual` for the `FACT_PROMPTS` keys and `creative` for the `STORY_PROMPTS` keys. That is 6 prompts × 3 versions.
  - `switch_points.csv` with columns: text, clean original, switch token, switch char, story tokens.
  - Word and token tables in `TEMPERATURE_WORDS_DIR` and `TEMPERATURE_TOKENS_DIR`, plus `run_log.csv`.
  - `marks.csv` with columns: text, word, char, type (`wrong fact` | `topic change` | `nonsense`), reason.

- [ ] **Step 1: Section 4, "Write the texts".**
  - For every prompt, write the cool version, the switch version (cool up to a random token in `SWITCH_RANGE`, then hot) and the hot version, all with `MIN_P`.
  - Seeds: `RANDOM_SEED + prompt number`. The switch draw uses `np.random.default_rng(RANDOM_SEED)`.
  - Skip the whole section when `switch_points.csv` exists. This mirrors notebook 03 section 3.

- [ ] **Step 2: Section 5, "Qwen reads every text".** Same as notebook 03 section 4, writing into the `TEMPERATURE_*` folders and skipping texts already read.

- [ ] **Step 3: Run notebook 05 in the background** with nbconvert `--execute --inplace`.
  - Expected: 18 `.txt` files, 18 word tables, 18 token tables, and a `run_log.csv` with 18 rows.
  - Every switch text matches its cool twin up to `switch char`. Check this in a cell: `switch_text[:char] == cool_text[:char]`.

- [ ] **Step 4: Read all 18 texts and write `marks.csv`.**
  - Mark every wrong fact, topic change or nonsense word. `char` is where the word starts in the text, checked with `text[char:char+len(word)] == word`.
  - Then tell the user that the marks are theirs to review.

### Task 3: The evolving distance to real writing (`analysis/evolving.py`)

**Files:**
- Create: `src/note_books/Qwen7B/analysis/evolving.py`, `src/note_books/Qwen7B/tests/test_evolving.py`
- Modify: `src/note_books/Qwen7B/analysis/__init__.py` (list the module in the docstring)

**Interfaces:**
- Consumes: `analysis.cluster.cluster(curves) -> (labels, centres, table)` and the `in_right_cluster` naming rule (each cluster is named after its majority kind).
- Produces:
  - `references_by_cluster(names: list[str], kinds: np.ndarray, curves: list[np.ndarray]) -> dict[str, dict[str, np.ndarray]]`: `{"factual": {name: curve}, "creative": {name: curve}}`. These are the 25 real texts grouped by the k-medoids cluster they fall in, each cluster named after its majority kind.
  - `distance_to_writing(curve: np.ndarray, references: dict[str, np.ndarray], window: int, leave_out: tuple[str, ...] = ()) -> np.ndarray`
    - Same length as `curve`.
    - The value at word t is the raw Euclidean distance between the stretch `curve[t-window+1 : t+1]` and the most similar stretch of any reference curve. It comes from `stumpy.aamp(curve, window, reference, ignore_trivial=False)`, taking the minimum over references and shifting so the index is the stretch's *last* word.
    - NaN for t < window − 1.
    - All NaN when `len(curve) < window`.
    - References shorter than `window`, or named in `leave_out`, are skipped.
    - Raises `ValueError` when no reference is left.
  - `evolving_profile(curve, references_by_kind, window, leave_out=()) -> pd.DataFrame`. Columns:
    - `word` (0..n−1)
    - `value` (the raw curve)
    - `to factual` and `to creative` (from `distance_to_writing`)
    - `to nearest`, the element-wise `np.fmin` of the two
    - A kind with no reference left after `leave_out` gets an all-NaN column instead of raising.
  - `normal_ceiling(real_curves: dict[str, np.ndarray], references_by_kind, window) -> float`: the highest `to nearest` value any real text reaches when it is left out of the references.
  - `first_alarm(to_nearest: np.ndarray, ceiling: float) -> int | None`: the first word whose value is above `ceiling` (NaN is never above), or None.

- [ ] **Step 1: Write the failing tests** in `tests/test_evolving.py`:

```python
import numpy as np
import pytest
from analysis.evolving import distance_to_writing, evolving_profile, normal_ceiling, first_alarm

BASE = np.array([1., 3., 2., 5., 4., 1., 2., 6., 3., 2.])

def test_identical_reference_gives_zero_after_the_first_window():
    d = distance_to_writing(BASE, {"a": BASE.copy()}, window=3)
    assert np.isnan(d[:2]).all() and np.allclose(d[2:], 0)

def test_spike_raises_only_stretches_that_contain_it():
    spiked = BASE.copy(); spiked[6] = 40.
    d = distance_to_writing(spiked, {"a": BASE}, window=3)
    assert (d[6:9] > 10).all() and np.allclose(d[2:6], 0) and np.isclose(d[9], 0)

def test_short_curve_is_all_nan():
    d = distance_to_writing(BASE[:2], {"a": BASE}, window=3)
    assert len(d) == 2 and np.isnan(d).all()

def test_short_reference_is_skipped():
    d = distance_to_writing(BASE, {"short": BASE[:2], "a": BASE}, window=3)
    assert np.allclose(d[2:], 0)

def test_no_reference_left_raises():
    with pytest.raises(ValueError):
        distance_to_writing(BASE, {"a": BASE}, window=3, leave_out=("a",))

def test_profile_leaves_out_named_texts():
    refs = {"factual": {"orig": BASE, "other": BASE + 10}, "creative": {"c": BASE + 20}}
    profile = evolving_profile(BASE, refs, window=3, leave_out=("orig",))
    assert list(profile.columns) == ["word", "value", "to factual", "to creative", "to nearest"]
    assert (profile["to factual"].iloc[2:] > 0).all()

def test_ceiling_leaves_each_text_out():
    zeros, ones = np.zeros(10), np.ones(10)
    real = {"x": zeros, "y": zeros.copy(), "z": ones}
    refs = {"factual": {"x": zeros, "y": zeros.copy()}, "creative": {"z": ones}}
    # x and y match each other (0); z, with itself left out, has no creative reference and is sqrt(3) from x and y
    assert normal_ceiling(real, refs, window=3) == pytest.approx(np.sqrt(3))

def test_profile_kind_with_no_reference_left_is_nan():
    refs = {"factual": {"x": np.zeros(10)}, "creative": {"z": np.ones(10)}}
    profile = evolving_profile(np.ones(10), refs, window=3, leave_out=("z",))
    assert profile["to creative"].isna().all() and np.allclose(profile["to nearest"].iloc[2:], np.sqrt(3))

def test_first_alarm_none_when_never_above():
    assert first_alarm(np.array([np.nan, 1., 2.]), 5.) is None

def test_first_alarm_ignores_nan():
    assert first_alarm(np.array([np.nan, 1., 7., 9.]), 5.) == 2
```

- [ ] **Step 2: Run** `python -m pytest tests/test_evolving.py -v`. Expected: FAIL with "No module named analysis.evolving".

- [ ] **Step 3: Implement the five functions** in `analysis/evolving.py`. The module docstring explains the idea in 4–6 lines, in the same style as `hallucinations.py`. `references_by_cluster` runs `cluster()` on the 25 curves; it has no unit test because Task 4 checks it against the known 23/25 result.

- [ ] **Step 4: Run** `python -m pytest tests -v`. Expected: all 13 tests pass (3 from Task 1, 10 here).

### Task 4: Figures — the evolving graph, static and animated

**Files:**
- Modify: `src/note_books/Qwen7B/analysis/figures.py`
- Create: `src/note_books/Qwen7B/tests/test_figures.py`

**Interfaces:**
- Consumes: Task 3's `evolving_profile` DataFrame columns, and the existing `save_to`, `clean_axes`, colour constants and `save_html` helpers.
- Produces:
  - `plot_evolving(title: str, profiles: dict[str, pd.DataFrame], ceilings: dict[str, float], real_nearest: dict[str, list[np.ndarray]], switch_word: dict[str, int | None], marks: dict[str, list[tuple[int, str]]]) -> Figure`. All dicts are keyed by signal name.
  - `animate_evolving(title: str, profile: pd.DataFrame, ceiling: float, switch_word: int | None, step: int = 5) -> plotly Figure`

- [ ] **Step 1: Write the failing tests**:

```python
import numpy as np, pandas as pd
from analysis.figures import plot_evolving, animate_evolving

def _profile(n=30):
    w = np.arange(n, dtype=float)
    return pd.DataFrame({"word": np.arange(n), "value": w, "to factual": w, "to creative": w + 1, "to nearest": w})

def test_plot_evolving_has_two_rows_per_signal():
    p = _profile()
    fig = plot_evolving("t", {"A": p, "B": p}, {"A": 5., "B": 5.}, {"A": [p["to nearest"].to_numpy()], "B": []},
                        {"A": 10, "B": None}, {"A": [(12, "London")], "B": []})
    assert len(fig.axes) == 4

def test_animation_has_one_frame_per_step():
    fig = animate_evolving("t", _profile(30), 5., 10, step=5)
    assert len(fig.frames) == 6
```

- [ ] **Step 2: Run** them. Expected: FAIL with an ImportError.

- [ ] **Step 3: Implement both figures.**
  - **`plot_evolving`:** for each signal, two stacked axes sharing x ("measured word, in reading order").
    - **Top axis:** the raw curve.
    - **Bottom axis:**
      - `to factual` in the factual blue and `to creative` in the creative orange.
      - Every real text's leave-one-out `to nearest` as thin grey lines (raw, no band).
      - The ceiling as a dashed grey horizontal line labelled "highest any real text reaches".
      - The first alarm as a red dot.
    - **On both axes:** the switch as a dashed vertical line, and marks as red vertical lines labelled with the word.
    - **Title:** the message, e.g. `"{text}: alarm at word 84, switch at word 71"`, or `"no alarm"`.
  - **`animate_evolving`:** a plotly figure with frames at words `step`, `2·step`, … (the last frame is always the full text; `save_html` takes no folder today, so save with `fig.write_html(TEMPERATURE_FIGURES_DIR / ...)`), showing the raw curve and the two distance lines up to that word, the ceiling line and the switch line. It has a play button and a slider. Save it with `save_html`, writing to the given folder.

- [ ] **Step 4: Run** `python -m pytest tests -v`. Expected: all 15 tests pass.

### Task 5: Notebook 06 — creative or hallucinating?

**Files:**
- Create: `src/note_books/Qwen7B/06_creative_or_hallucinating.ipynb`
- Output: `results/temperature/tables/{alarms.csv,marks_on_curve.csv}`, and `results/temperature/figures/evolving_{text}.png` plus `.html`

**Interfaces:**
- Consumes:
  - Task 3: `references_by_cluster`, `evolving_profile`, `normal_ceiling`, `first_alarm`
  - Task 4: `plot_evolving`, `animate_evolving`
  - Existing: `load_tables`, `curves_of`, `curve_with_rows`, `signal_by_key`, `dataset_names`, `kinds_of`, `planted_texts`, `row_at_char`, `first_row_from_char`, `curve_position`
- Texts compared:
  - the 18 temperature texts
  - the 8 planted texts, which have exact answer keys for wrong facts (e.g. Cyprus → Crete), with `leave_out=(clean original,)`

- [ ] **Step 1: Build the notebook with these sections.** Each has a short markdown cell saying what it shows.
  1. **Setup and loading:** the real tables, the temperature tables, and the planted tables from `HALLUCINATED_*`.
  2. **The reference writing.** For each signal in `EVOLVING_SIGNALS`, run `references_by_cluster` and print the cluster sizes and the number of texts with their own kind (it must be 23 of 25 for `glove_top5`). Also compute `normal_ceiling` and print it.
  3. **One evolving graph per text:** `plot_evolving` plus `animate_evolving`, saved to `TEMPERATURE_FIGURES_DIR`.
    - Switch words come from `switch_points.csv` via `first_row_from_char` and `curve_position`.
    - Marks come from `marks.csv` for the temperature texts and from the planted edits for the planted texts.
  4. **The alarm table** (`alarms.csv`), one row per text and signal, with these columns:
    - text, kind, made as (`cool` | `switch` | `hot` | `planted`), signal
    - words on curve, switch word, first alarm word
    - alarms before switch: words above the ceiling before the switch
    - alarms after switch: the same count after it
    - words above ceiling: the count for the whole text
  5. **Marks on the curve** (`marks_on_curve.csv`): for every mark and planted edit, its `to nearest` value, whether it is above the ceiling, and its rank in its own text's `to nearest` curve (`rank_in_curve`).
  6. **The answer** (markdown, written after the run, from the tables only). Three questions:
    - Do the cool creative stories stay below the ceiling?
    - Do the switch and hot texts cross it, and after the switch?
    - Does a factual article lean toward creative (`to creative` < `to factual`) after the switch?

- [ ] **Step 2: Run** notebook 06 with nbconvert `--execute --inplace`. Expected:
  - no errors
  - `alarms.csv` has 52 rows (26 texts × 2 signals)
  - a PNG and an HTML file for each of the 26 texts

- [ ] **Step 3: Check the figures.** Open 3 of them (one cool creative, one factual switch, one planted), make sure the labels are readable and nothing overlaps, and send them to the user.

### Task 6: Write it down

**Files:**
- Modify: `GLOSSARY.md`, `src/note_books/Qwen7B/CONCLUSIONS.md`

- [ ] **Step 1: `GLOSSARY.md`.** Add these terms, with no implementation details:
  - **Evolving graph:** a text's curve drawn as it is read, with its distance to real factual and creative writing at every word.
  - **Distance to real writing:** how unlike the most similar stretch of any real text of a kind the last few words are.
  - **Normal ceiling:** the highest distance to real writing any real text reaches when it isn't compared with itself.
  - **Alarm:** a word where the distance goes above the normal ceiling.
  - **Cool / hot / switch text:** written at the low temperature, at the high temperature, or low then high from a random point.

- [ ] **Step 2: `CONCLUSIONS.md`.** Add a section, "Creative or hallucinating (temperature)", in the existing 5-part style: what was checked, how, the results (the alarm table plus 2 figures), what we can gather, and how it ran in code. Every number must come from `alarms.csv` or `marks_on_curve.csv`; check each one against the CSV.
