from cornsnake import util_dir
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from csfy import config, util_config

from . import util_labels

class State:
    def __init__(self, model, tokenizer, label_mappings) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_mappings = label_mappings

    def is_onnx(self):
        return False

def load_model_state(path_to_model=None):
    path_to_model = path_to_model if path_to_model else util_config.path_to_checkpoint()
    tokenizer = DistilBertTokenizer.from_pretrained(config.TOKENIZER)
    model = DistilBertForSequenceClassification.from_pretrained(path_to_model)

    label_mappings = util_labels.load_label_mapping(util_dir.get_parent_dir(path_to_model))

    model.eval()
    return State(model, tokenizer, label_mappings)
