import click
from cornsnake import decorators

from csfy.cli import pass_environment
from csfy.core import export_onnx

@click.command("export", short_help="Exports a model previously created via the 'train' command, to ONNX format.")
@click.argument("path_to_model", required=False, type=click.Path(resolve_path=True))
@click.argument("output", required=True, type=click.Path(resolve_path=True))
@pass_environment
@decorators.timer
def cli(ctx, path_to_model, output):
    """Export a previously trained model to ONNX format"""
    export_onnx.export_to_onnx(path_to_model, output)
    ctx.log(f"Exported the model to {click.format_filename(output)}")
