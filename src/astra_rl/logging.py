import logging
from typing import Any, Dict
import os

is_wandb_installed: bool = False
try:
    import wandb

    is_wandb_installed = True
except ImportError:
    logging.warning("Wandb not installed/found in path.")
    is_wandb_installed = False


def config_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("astra").setLevel(logging.DEBUG)


logger = logging.getLogger("astra")
config_logging()


class ASTRAWandbLogger:
    """Wandb logger class for ASTRA RL.

    Checks if wandb is installed and needs WANDB_API_KEY set in the environment.

    Attributes:
        run (wandb.Run): The Weights & Biases run object.
    """

    def __init__(self, wandb_kwargs: Dict[str, Any]) -> None:
        """Initialize the Weights & Biases logger.

        Args:
            wandb_kwargs (Dict[str, Any]): Keyword arguments for configuring Weights & Biases.
        """

        if not is_wandb_installed:
            raise ImportError(
                "Wandb not installed. Install it with `pip install wandb` or `pip install astra_rl[wandb]` to use Weights & Biases logging."
            )

        if "WANDB_API_KEY" not in os.environ:
            raise EnvironmentError(
                "WANDB_API_KEY environment variable is not set. Please set it to use Weights & Biases."
            )
        self.run = wandb.init(project="astra_rl", config=wandb_kwargs)

    def log(self, current_logs: Dict[Any, Any]) -> None:
        """Log the current step to Weights & Biases.

        Args:
            current_logs (Dict[Any, Any]): A dictionary containing the logs for the current step.
        """

        self.run.log(current_logs)
