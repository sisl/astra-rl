import typer

# Import commands & sub-commands
from astra_rl.cli.train import app as train_cmd
from astra_rl.cli.evaluate import app as evaluate_cmd

app = typer.Typer()

# Directly add sub-commands to the main app
app.add_typer(train_cmd)
app.add_typer(evaluate_cmd)


# Entry point function needed by the package installation
def main() -> None:
    """
    ASTRA-RL CLI - A command line interface for the ASTRA-RL toolbox.
    """
    app()
