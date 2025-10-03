"""
trainer.py
The trainer is an opinionated system designed for making training new models easy. To gain full customization over the model training pipeline, we recommend using the lower-level `Harness` system in `harness.py`.
"""

import torch
from typing import Generic
from pydantic import BaseModel
from torch.optim import Optimizer

from astra_rl.training.harness import Harness
from astra_rl.core.sampler import Sampler
from astra_rl.core.algorithm import Algorithm
from astra_rl.core.common import ActionT, StateT, Batch, Step


class TrainingConfiguration(BaseModel):
    """A typechecked dataclass which configures the training procedure.

    Attributes:
        lr (float): Learning rate for the optimizer.
        batch_size (int): Size of each batch (after flattening from experience) for training.
        optimizer (str): Type of optimizer to use [choices: "adam", "adamw", "sgd", "rmsprop", "adagrad"].
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating the model weights.
        training_steps (int): Total number of rollouts to run and train for.
        num_episodes_per_experience (int): Number of rollouts to run before making a gradient update.
    """

    # optimization configuration
    lr: float = 3e-3
    batch_size: int = 16
    optimizer: str = "adamw"
    gradient_accumulation_steps: int = 1  # how many

    # training configuration
    training_steps: int = 1024  # how many rollouts to train for

    # rollout configuration
    num_episodes_per_experience: int = 8  # how many rollouts per gradient update


class Trainer(Generic[StateT, ActionT, Step, Batch]):
    """A high-level trainer that pushbutton trains your policy

    Example:
        Here is an example of how to use the `Trainer` class with the DPO algorithm
        and an AST problem sampler

        >>> import torch
        >>> from astra_rl import (
        ...     Trainer,
        ...     TrainingConfiguration,
        ... )
        >>> from astra_rl.algorithms.dpo import (
        ...     DPO,
        ... )
        >>> from astra_rl.methods.ast import (
        ...     ASTProblem,
        ...     ASTSampler,
        ... )
        >>>
        >>> problem = (
        ...     ASTProblem()
        ... )
        >>> sampler = (
        ...     ASTSampler(
        ...         problem, ...
        ...     )
        ... )
        >>> algorithm = DPO(...)
        >>> config = TrainingConfiguration(
        ...     lr=1e-3,
        ...     batch_size=16,
        ...     optimizer="adamw",
        ...     gradient_accumulation_steps=1,
        ...     training_steps=1024,
        ...     num_episodes_per_experience=8,
        ... )
        >>> trainer = Trainer(
        ...     config,
        ...     sampler,
        ...     algorithm,
        ... )
        >>> trainer.train()

    Attributes:
        config (TrainingConfiguration): The configuration for the training process.
        harness (Harness): The harness that manages the training loop and interactions with the sampler. See `astra_rl.training.harness` for what it does.
        optimizer (Optimizer): The optimizer used for updating the model parameters.
        _global_step_counter (int): A counter for global steps, used for gradient accumulation.
    """

    optimizer: Optimizer

    def __init__(
        self,
        config: TrainingConfiguration,
        sampler: Sampler[StateT, ActionT],
        algorithm: Algorithm[StateT, ActionT, Step, Batch],
    ):
        """
        Args:
            config (TrainingConfiguration): The configuration for the training process.
            sampler (Sampler): The sampler to run our algorithm in.
            algorithm (Algorithm): The algorithm used for training the tester agent.
        """

        self.config = config
        self.harness = Harness(sampler, algorithm, config.num_episodes_per_experience)

        # TODO initialize LR scheduler?
        # ?????????????????????????????

        # initialize optimizer
        if config.optimizer == "adam":
            from torch.optim import Adam

            self.optimizer = Adam(sampler.system.parameters(), config.lr)
        elif config.optimizer == "adamw":
            from torch.optim import AdamW

            self.optimizer = AdamW(sampler.system.parameters(), config.lr)
        elif config.optimizer == "sgd":
            from torch.optim import SGD

            self.optimizer = SGD(sampler.system.parameters(), config.lr)
        elif config.optimizer == "rmsprop":
            from torch.optim import RMSprop

            self.optimizer = RMSprop(sampler.system.parameters(), config.lr)
        elif config.optimizer == "adagrad":
            from torch.optim import Adagrad

            self.optimizer = Adagrad(sampler.system.parameters(), config.lr)
        else:
            raise ValueError(f"Unknown optimizer configured: {config.optimizer}")

        # step counter, for acccmulutaion, etc.
        self._global_step_counter = 0

    def train(self) -> None:
        """Run training by the specified config!

        Note:
            This method takes no arguments and returns nothing, and its
            only used for side effects. We don't really need it other than
            it's helpful for allowing the user to control when training
            actually starts (instead of immediately after Trainer construction).
        """
        for _ in range(self.config.training_steps):
            buf = self.harness.experience()
            for batch in buf:
                # increment counter first for occumulation
                self._global_step_counter += 1
                loss: torch.Tensor = (
                    self.harness.step(batch)[0]
                    / self.config.gradient_accumulation_steps
                )
                # typing disabled here b/c mypy can't statically verify
                # that the loss has gradients
                loss.backward()  # type: ignore[no-untyped-call]

                # if gradient accumulation happens, step!
                if (
                    self._global_step_counter % self.config.gradient_accumulation_steps
                    == 0
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
