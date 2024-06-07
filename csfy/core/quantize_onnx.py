from enum import Enum

from onnxruntime.quantization.quantize import quantize_dynamic, QuantType


class QuantizationLevel(str, Enum):
    # TODO - q_4 = "q_4"
    q_8 = "q_8"
    q_u8 = "q_u8"
    q_f8 = "q_f8"
    q_16 = "q_16"
    q_u16 = "q_u16"

def get_quantization_levels():
    return [e.value for e in QuantizationLevel]

def _get_quantization(quantization_level):
    # ref LATEST = https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quant_utils.py
    match quantization_level:
        # TODO - enable this when MS release
        # case QuantizationLevel.q_4:
        #     return QuantType.QInt4
        case QuantizationLevel.q_8:
            return QuantType.QInt8
        case QuantizationLevel.q_u8:
            return QuantType.QUInt8
        case QuantizationLevel.q_f8:
            return QuantType.QFLOAT8E4M3FN
        case QuantizationLevel.q_16:
            return QuantType.QInt16
        case QuantizationLevel.q_u16:
            return QuantType.QUInt16
        case _:
            valid_values = ",".join(get_quantization_levels())
            raise ValueError(f"Not a recognised quantization level: {quantization_level}. Valid values are: [{valid_values}]")

def quantize(path_to_onnx_input, path_to_onnx_output, quantization_level):
    quantization = _get_quantization(quantization_level)

    quantized_model = quantize_dynamic(
        model_input=path_to_onnx_input,
        model_output=path_to_onnx_output,
        weight_type=quantization
    )
