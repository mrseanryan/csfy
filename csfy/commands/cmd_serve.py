import click

from csfy.cli import pass_environment
from csfy.api import main

from csfy import config
from csfy.core import predictor

@click.command("serve", short_help=f"Serve model via a REST API that can accept requests to predict a label for given text.")
@click.argument("path_to_model", required=True, type=click.Path(resolve_path=True))
@pass_environment
def cli(ctx, path_to_model):
    config.SERVE_MODEL_PATH = path_to_model

    state = predictor.load_state(path_to_model)
    predictor.cached_state = state

    main.start()
