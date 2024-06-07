from cornsnake import util_dir
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizer

from csfy import config
from . import util_labels

class OnnxState:
    def __init__(self, tokenizer, ort_session, label_mappings) -> None:
        self.tokenizer = tokenizer
        self.ort_session = ort_session
        self.label_mappings = label_mappings

    def is_onnx(self):
        return True

def load_state(path_to_onnx):
    tokenizer = DistilBertTokenizer.from_pretrained(config.BASE_MODEL)
    ort_session = ort.InferenceSession(path_to_onnx)
    label_mappings = util_labels.load_label_mapping(util_dir.get_parent_dir(path_to_onnx))
    return OnnxState(tokenizer, ort_session, label_mappings)


def predict_via_onnx(text, state):
    model_expected_input_shape = state.ort_session.get_inputs()[0].shape
    print("Model expects input shape:", model_expected_input_shape)
    inputs = state.tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=model_expected_input_shape[1])
    print("input shape", inputs['input_ids'].shape)

    input_ids = inputs['input_ids']
    if input_ids.ndim == 1:
        input_ids = input_ids[np.newaxis, :]
    ort_inputs = {state.ort_session.get_inputs()[0].name: input_ids}

    ort_inputs['input_ids'] = ort_inputs['input_ids'].astype(np.int64)

    ort_outputs = state.ort_session.run(None, ort_inputs)
    predictions = np.argmax(ort_outputs, axis=-1)

    predicted_label = state.label_mappings[predictions.item()]
    return predicted_label
