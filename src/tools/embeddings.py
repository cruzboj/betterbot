def get_bert_vec(text, bert_model, bert_tokenizer , model):
    # המרת הטקסט למספרי טוקנים של BERT
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # שליפת השכבה האחרונה: צורה של (1, num_tokens, 768)
        hidden_states = outputs.last_hidden_state
        
        # חיתוך [CLS] בהתחלה (0) ו-[SEP] בסוף (-1), וחישוב ממוצע על טוקני המילה/המשפט
        vec = hidden_states[0, 1:-1, :].mean(dim=0)
        return vec