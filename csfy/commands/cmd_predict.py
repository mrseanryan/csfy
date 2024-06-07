import click
from cornsnake import decorators, util_input, util_print
import os

from csfy.cli import pass_environment
from csfy.core import predict_onnx, predict_pytorch


def _is_onnx(path_to_file):
    ext = os.path.splitext(path_to_file)[1]
    return ext == ".onnx"

def _load_state(path_to_model):
    if _is_onnx(path_to_model):
        return predict_onnx.load_state(path_to_model)
    else:
        return predict_pytorch.load_state(path_to_model)

@decorators.timer
def _predict(text, state):
    if state.is_onnx():
        label = predict_onnx.predict_via_onnx(text, state)
        return label
    else:
        return predict_pytorch.predict(text, state)

def _chat(state):
    USER_EXIT = "bye"

    while True:
        user_query = util_input.input_required("How can I help? [to exit, type 'bye' and press ENTER] >>", "")
        if user_query.lower() == USER_EXIT.lower():
            print("Goodbye for now")
            break
        if not user_query:
            continue
        label = _predict(text=user_query, state=state)
        util_print.print_result(label)

@click.command("predict", short_help="Predicts a labal for the given text, using a model previously created via the 'train' command.")
@click.argument("path_to_model", required=False, type=click.Path(resolve_path=True))
@click.argument("text", required=True, type=click.STRING)
@click.option('--chat', '-c', is_flag=True, help="Enable chat mode (interactive loop).")
@pass_environment
def cli(ctx, path_to_model, text, chat):
    """Predicts a labal for the given text, using a model previously created via the 'train' command."""
    state = _load_state(path_to_model)
    label = _predict(text, state)
    # TODO improve logging
    ctx.log(f"Predicted label: '{label}' - for '{text}'")
    if chat:
        _chat(state)
