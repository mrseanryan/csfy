from enum import Enum

from onnxruntime.quantization.quantize import quantize_dynamic, QuantType


class QuantizationLevel(str, Enum):
    q_4 = "q_4"
    q_8 = "q_8"

def _get_quantization(quantization_level):
    match quantization_level:
        case QuantizationLevel.q_4:
            return QuantType.QInt4
        case QuantizationLevel.q_8:
            return QuantType.QInt8
        case _:
            raise ValueError(f"Not a recognised quantization level: {quantization_level}")

def quantize(path_to_onnx_input, path_to_onnx_output, quantization_level):
    quantization = _get_quantization(quantization_level)

    quantized_model = quantize_dynamic(
        model_input=path_to_onnx_input,
        model_output=path_to_onnx_output,
        weight_type=quantization
    )
