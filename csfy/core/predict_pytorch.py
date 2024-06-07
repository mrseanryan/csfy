import torch

from . import util_model

def load_state(path_to_model):
    state = util_model.load_model_state(path_to_model)
    return state

def _tokenize_text(text, state):
    return state.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

def predict(text, state):
    tokenized_text = _tokenize_text(text, state)
    with torch.no_grad():
        outputs = state.model(**tokenized_text)
        predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_label = state.label_mappings[predictions.item()]
    return predicted_label
