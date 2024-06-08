import os
from cornsnake import decorators, zip_dir, util_dir
import click

from csfy.cli import pass_environment

def _get_files_to_include_avoiding_old_checkpoints(path_to_model_dir):
    files_to_include = []

    # If there is a final trained model, then skip all checkpoints
    has_final = os.path.exists(os.path.join(path_to_model_dir, "trained.model"))
    if has_final:
        print("Found final trained model - so excluding checkpoints")
    # else take the most recent checkpoint
    max_checkpoint = 0
    checkpoint_id_to_keep = None

    all_files = util_dir.find_files_recursively(path_to_model_dir)

    if not has_final:
        for file in all_files:
            if "checkpoint-" in file:
                dir_parts = util_dir.get_dir_parts(file)
                checkpoint_part = list(filter(lambda p: "checkpoint-" in p, dir_parts))[0]
                parts = checkpoint_part.split("checkpoint-")
                last_part = parts[-1]
                checkpoint_id = int(last_part)
                if checkpoint_id > max_checkpoint:
                    checkpoint_id_to_keep = checkpoint_id
                    max_checkpoint = checkpoint_id

    for file in all_files:
        if "checkpoint-" in file:
            if has_final:
                continue
            if f"checkpoint-{checkpoint_id_to_keep}" in file:
                files_to_include.append(file)
        else:
            files_to_include.append(file)

    return files_to_include

@click.command("zip", short_help="ZIP a trained model for easier distribution.")
@click.argument("path_to_model_dir", required=True, type=click.Path(resolve_path=True))
@click.argument("path_to_output_zipfile", required=True, type=click.Path(resolve_path=True))
@pass_environment
@decorators.timer
def cli(ctx, path_to_model_dir, path_to_output_zipfile):
    ctx.log(f"ZIPping model at {click.format_filename(path_to_model_dir)}")
    files_to_include = _get_files_to_include_avoiding_old_checkpoints(path_to_model_dir)

    file_names_in_zip = zip_dir.create_zip_of_files(files_to_include, path_to_model_dir, path_to_output_zipfile)
    print(f" - {file_names_in_zip}")

    ctx.log(f"ZIP created at {click.format_filename(path_to_output_zipfile)}")
