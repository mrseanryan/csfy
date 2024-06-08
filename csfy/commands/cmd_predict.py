import click
from cornsnake import util_input, util_print

from csfy.core import predictor
from csfy.cli import pass_environment

def _chat(state):
    USER_EXIT = "bye"

    while True:
        user_query = util_input.input_required("How can I help? [to exit, type 'bye' and press ENTER] >>", "")
        if user_query.lower() == USER_EXIT.lower():
            print("Goodbye for now")
            break
        if not user_query:
            continue
        label = predictor.predict_label(text=user_query, state=state)
        util_print.print_result(label)

@click.command("predict", short_help="Predicts a labal for the given text, using a model previously created via the 'train' command.")
@click.argument("path_to_model", required=False, type=click.Path(resolve_path=True))
@click.argument("text", required=True, type=click.STRING)
@click.option('--chat', '-c', is_flag=True, help="Enable chat mode (interactive loop).")
@pass_environment
def cli(ctx, path_to_model, text, chat):
    """Predicts a labal for the given text, using a model previously created via the 'train' command."""
    state = predictor.load_state(path_to_model)
    label = predictor.predict_label(text, state)
    # TODO improve logging
    ctx.log(f"Predicted label: '{label}' - for '{text}'")
    if chat:
        _chat(state)
