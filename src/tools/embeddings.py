import torch


def get_bert_vec(text, bert_tokenizer, bert_model):
    inputs = bert_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(bert_model.device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        hidden_states = outputs.last_hidden_state  # (1, num_tokens, 768)

        if hidden_states.shape[1] > 2:
            vec = hidden_states[0, 1:-1, :].mean(dim=0)
        else:
            vec = hidden_states[0].mean(dim=0)

    return vec