"""Writing the clean texts and their hot-burst copies (notebook 03, needs the GPU).

The next token is drawn from softmax(logits / temperature) (Qwen's default top-k, top-p and repetition penalty are
switched off), keeping only tokens with at least min_p of the #1 token's probability. Without min_p, temperatures
above ~1.15 break Qwen into mixed scripts and code; with it, a hot Qwen gets things wrong while it still writes
sentences. The cool and hot parts of a text use the same min_p, so temperature is the only thing that changes.
"""
import torch

from qwen_config import BURST_RANGE, GENERATION_MAX_TOKENS, STORY_WORDS


def request_text(kind, prompt):
    """What Qwen is asked to write: a story ("creative") or an encyclopedia article ("factual") about `prompt`."""
    if kind == "creative":
        return (f"Write a story of about {STORY_WORDS} words about {prompt}. "
                "Write only the story itself: no title, no headings, no notes.")
    if kind == "factual":
        return (f"Write an encyclopedia article of about {STORY_WORDS} words about {prompt}. "
                "Write only the article itself: no title, no headings, no notes.")
    raise ValueError(f"kind must be 'creative' or 'factual', not {kind!r}")


def story_request(tokenizer, prompt, device, kind="creative"):
    """The chat-formatted request for one text, as token ids."""
    messages = [{"role": "user", "content": request_text(kind, prompt)}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True)
    return encoded["input_ids"].to(device)


def write_story(model, tokenizer, prompt, temperature, seed, story_start_ids=None, max_new_tokens=GENERATION_MAX_TOKENS,
                kind="creative", min_p=0.0):
    """The text's token ids (special tokens removed). With `story_start_ids`, the text is continued from those
    tokens (they are included in what is returned). kind: "creative" (story) or "factual" (article)."""
    request = story_request(tokenizer, prompt, model.device, kind)
    if story_start_ids is None:
        story_start_ids = []
    start = torch.tensor([story_start_ids], dtype=request.dtype, device=model.device)
    inputs = torch.cat([request, start], dim=1)
    torch.manual_seed(seed)
    output = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=True, temperature=temperature,
                            top_k=0, top_p=1.0, min_p=min_p, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id)
    new_ids = output[0, request.shape[1]:].tolist()
    special = set(tokenizer.all_special_ids)
    story_ids = []
    for token_id in new_ids:
        if token_id not in special:
            story_ids.append(token_id)
    return story_ids


def write_story_streaming(model, tokenizer, prompt, temperature, seed, story_start_ids=None,
                          max_new_tokens=GENERATION_MAX_TOKENS, kind="creative", min_p=0.0):
    """Same as write_story (same settings, same seed -> same text), but prints the text while Qwen writes it:
    first the given `story_start_ids`, if any, then each new piece as soon as it is drawn."""
    from transformers import TextStreamer

    request = story_request(tokenizer, prompt, model.device, kind)
    if story_start_ids is None:
        story_start_ids = []
    if story_start_ids:
        print(tokenizer.decode(story_start_ids, skip_special_tokens=True), end="", flush=True)
    start = torch.tensor([story_start_ids], dtype=request.dtype, device=model.device)
    inputs = torch.cat([request, start], dim=1)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)   # skip_prompt: not the request
    torch.manual_seed(seed)
    output = model.generate(inputs, attention_mask=torch.ones_like(inputs), do_sample=True, temperature=temperature,
                            top_k=0, top_p=1.0, min_p=min_p, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id, streamer=streamer)
    new_ids = output[0, request.shape[1]:].tolist()
    special = set(tokenizer.all_special_ids)
    story_ids = []
    for token_id in new_ids:
        if token_id not in special:
            story_ids.append(token_id)
    return story_ids


def draw_burst_start(story_length, rng):
    """A random token position within BURST_RANGE of the story's length."""
    low = int(BURST_RANGE[0] * story_length)
    high = int(BURST_RANGE[1] * story_length)
    return int(rng.integers(low, high + 1))


def burst_story(model, tokenizer, prompt, cool_ids, seed, burst_start, burst_tokens, cool_temperature,
                hot_temperature, kind="creative", min_p=0.0):
    """(text token ids, (burst start, burst end) tokens): the clean original `cool_ids` up to `burst_start`, then
    `burst_tokens` tokens at `hot_temperature`, then back to `cool_temperature` until the text is as long as the
    clean original. One short hot stretch keeps the text readable: a wrong fact or a turn, not word salad."""
    hot_ids = write_story(model, tokenizer, prompt, hot_temperature, seed=seed + 2000,
                          story_start_ids=cool_ids[:burst_start], max_new_tokens=burst_tokens, kind=kind, min_p=min_p)
    burst_end = len(hot_ids)
    story_ids = write_story(model, tokenizer, prompt, cool_temperature, seed=seed + 3000, story_start_ids=hot_ids,
                            max_new_tokens=max(len(cool_ids) - burst_end, 1), kind=kind, min_p=min_p)
    return story_ids, (burst_start, burst_end)


def decode(tokenizer, ids):
    """Token ids -> text."""
    return tokenizer.decode(ids, skip_special_tokens=True)
