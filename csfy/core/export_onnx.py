import torch

from csfy import config
from . import util_model

def export_to_onnx(path_to_model, path_to_onnx):
    state = util_model.load_model_state(path_to_model)

    input_ids = state.tokenizer(config.EXAMPLE_USER_PROMPT, return_tensors="pt").input_ids

    torch.onnx.export(state.model,
                    input_ids,
                    path_to_onnx,
                    export_params=True,  # Include the trained parameter weights inside the model file
                    opset_version=11,    # ONNX opset version
                    do_constant_folding=True,  # Execute constant folding for optimization
                    input_names = ['input_ids'],    # The model's input labels
                    output_names = ['output'],      # The model's output labels
                    dynamic_axes={'input_ids': {0: 'batch_size'},  # dynamic axes (variable length)
                                    'output': {0: 'batch_size'}})
    print(f"Exported ONNX to {path_to_onnx}")
