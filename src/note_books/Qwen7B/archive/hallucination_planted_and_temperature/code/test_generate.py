import pytest

from analysis.generate import request_text


def test_creative_request_is_a_story():
    text = request_text("creative", "a lost wand")
    assert "story" in text and "a lost wand" in text and "650" in text


def test_factual_request_is_an_article_of_facts():
    text = request_text("factual", "Paris, the capital of France")
    assert "encyclopedia" in text and "Paris, the capital of France" in text and "650" in text
    assert "story" not in text


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        request_text("poem", "x")


class _FakeTokenizer:
    all_special_ids = [99]
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        import torch
        return {"input_ids": torch.tensor([[7, 7, 7]])}


class _FakeModel:
    """Writes token `temperature * 10` every step, so each part of a text shows the temperature it was written at."""
    device = "cpu"

    def generate(self, inputs, temperature, max_new_tokens, **kwargs):
        import torch
        new = torch.full((1, max_new_tokens), int(temperature * 10), dtype=inputs.dtype)
        return torch.cat([inputs, new], dim=1)


def test_burst_story_is_cool_then_hot_then_cool_again():
    from analysis.generate import burst_story
    cool_ids = [1] * 20
    ids, burst = burst_story(_FakeModel(), _FakeTokenizer(), "x", cool_ids, seed=0, burst_start=5, burst_tokens=4,
                             cool_temperature=0.7, hot_temperature=2.2)
    assert burst == (5, 9)
    assert ids[:5] == [1] * 5            # the clean original up to the burst
    assert ids[5:9] == [22] * 4          # the hot burst
    assert ids[9:] == [7] * 11           # cool again, as long as the clean original
    assert len(ids) == len(cool_ids)
