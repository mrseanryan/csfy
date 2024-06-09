import os
import sys

import click

from cornsnake import util_log

from . import util_config

CONTEXT_SETTINGS = dict(auto_envvar_prefix="CSFY")

class Environment:
    def __init__(self):
        self.verbose = False

    def log(self, msg, *args):
        """Logs a message to stderr."""
        if args:
            msg %= args
        click.echo(msg, file=sys.stderr)

    def vlog(self, msg, *args):
        """Logs a message to stderr only if verbose is enabled."""
        if self.verbose:
            self.log(msg, *args)


pass_environment = click.make_pass_decorator(Environment, ensure=True)
cmd_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "commands"))

class CsfyCLI(click.Group):
    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(cmd_folder):
            if filename.endswith(".py") and filename.startswith("cmd_"):
                command = filename[4:-3]
                rv.append(command)
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        try:
            mod = __import__(f"csfy.commands.cmd_{name}", None, None, ["cli"])
        except ImportError as ie:
            util_log.log_exception(ie)
            return None
        return mod.cli


@click.command(cls=CsfyCLI, context_settings=CONTEXT_SETTINGS)
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode.")
@pass_environment
def start(ctx, verbose):
    """csfy (classify) is a command line tool to train and run simple text based classifiers.

    - for help about each command, add --help. for example:

        csfy train --help
    """
    ctx.verbose = verbose
    util_config.populate_config()

if __name__ == "__main__":
    start()
