"""The verbatim test: has the model memorised a text?

The model is given MEMORISATION_PROMPT_TOKENS tokens of the real text and writes the next
MEMORISATION_CONTINUATION_TOKENS tokens by itself, always taking its #1 choice. The match score is the share of
those tokens that equal the real text at the same position. This is done for MEMORISATION_PASSAGES passages spread
evenly through the text; the text's score is their average. A text whose score reaches MEMORISATION_THRESHOLD is
memorised: the model can reproduce it, so its signals reflect recall rather than reading.

Unlike "how often is the #1 guess right", this doesn't confuse memorised with merely predictable: a predictable
factual text gets single words right, but the model can't write 50 tokens of it word for word.
"""
import numpy as np
import pandas as pd

from config import (MEMORISATION_CONTINUATION_TOKENS, MEMORISATION_PASSAGES, MEMORISATION_PROMPT_TOKENS,
                    MEMORISATION_THRESHOLD)


def passage_starts(text_length):
    """Token positions where each tested continuation starts, spread evenly through the text."""
    first = MEMORISATION_PROMPT_TOKENS
    last = text_length - MEMORISATION_CONTINUATION_TOKENS
    if last < first:
        raise ValueError(f"text too short for the test: {text_length} tokens")
    starts = []
    for position in np.linspace(first, last, MEMORISATION_PASSAGES):
        starts.append(int(round(position)))
    return starts


def continuation_match(model, tokenizer, token_ids, start):
    """Share of the model's own continuation (from `start`) that equals the real text, token by token."""
    import torch

    prompt = torch.tensor([token_ids[start - MEMORISATION_PROMPT_TOKENS:start]], device=model.device)
    with torch.no_grad():
        output = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                max_new_tokens=MEMORISATION_CONTINUATION_TOKENS, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    written = output[0, prompt.shape[1]:].tolist()
    real = token_ids[start:start + MEMORISATION_CONTINUATION_TOKENS]
    matches = 0
    for position in range(len(real)):
        if position < len(written) and written[position] == real[position]:
            matches += 1
    return matches / len(real)


def memorisation_scores(model, tokenizer, texts):
    """One row per text: its passage match scores, their average, and whether it counts as memorised."""
    rows = []
    for name, text in texts.items():
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        passage_scores = []
        for start in passage_starts(len(token_ids)):
            passage_scores.append(continuation_match(model, tokenizer, token_ids, start))
        average = float(np.mean(passage_scores))
        row = {"text": name, "match score": average, "best passage": max(passage_scores)}
        for number, passage_score in enumerate(passage_scores, start=1):
            row[f"passage {number}"] = passage_score
        row["memorised"] = average >= MEMORISATION_THRESHOLD
        rows.append(row)
    return pd.DataFrame(rows).sort_values("match score", ascending=False).reset_index(drop=True)
