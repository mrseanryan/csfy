from cornsnake import decorators
import click

from csfy.cli import pass_environment
from csfy.core import train_classifier

@click.command("train", short_help="Trains a model to classify text, predicting a label.")
@click.argument("path_to_data", required=True, type=click.Path(resolve_path=True))
@pass_environment
@decorators.timer
def cli(ctx, path_to_data):
    """Train a model from the labelled data in a parquet file."""
    ctx.log(f"Training model from data at {click.format_filename(path_to_data)}")
    train_classifier.train(path_to_data)
