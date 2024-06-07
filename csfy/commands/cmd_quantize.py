import click
from cornsnake import decorators

from csfy.cli import pass_environment
from csfy.core import quantize_onnx

@click.command("quantize", short_help="Quantize an existing ONNX model to reduce size and inference time whilst mostly preserving accuracy. The quantization level can be 'q_4' or 'q_8'.")
@click.argument("input_onnx", required=True, type=click.Path(resolve_path=True))
@click.argument("output_onnx", required=True, type=click.Path(resolve_path=True))
@click.argument("quantization_level", required=True, type=click.STRING)
@pass_environment
@decorators.timer
def cli(ctx, input_onnx, output_onnx, quantization_level):
    """Quantize an ONNX file to a smaller, slightly less accurate model"""
    quantize_onnx.quantize(input_onnx, output_onnx, quantization_level)
    ctx.log(f"Exported the quantized model to {click.format_filename(output_onnx)}")
