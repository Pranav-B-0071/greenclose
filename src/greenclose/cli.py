"""GREENCLOSE command line. Paper only: nothing here can place an order."""

from __future__ import annotations

import typer

from greenclose import __version__
from greenclose.ops.logging import configure

app = typer.Typer(
    name="greenclose",
    help="GREENCLOSE research instrument. PAPER ONLY. No broker, no orders, no money.",
    no_args_is_help=True,
)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    configure("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
