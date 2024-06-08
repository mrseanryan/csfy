from cornsnake import decorators
import os

from csfy.core import predict_onnx, predict_pytorch

def _is_onnx(path_to_file):
    ext = os.path.splitext(path_to_file)[1]
    return ext == ".onnx"

def load_state(path_to_model):
    if _is_onnx(path_to_model):
        return predict_onnx.load_state(path_to_model)
    else:
        return predict_pytorch.load_state(path_to_model)

cached_state = None

@decorators.timer
def predict_label(text, state=None):
    state = state if state else cached_state
    if state.is_onnx():
        label = predict_onnx.predict_via_onnx(text, state)
        return label
    else:
        return predict_pytorch.predict(text, state)
