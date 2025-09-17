from typing import Generic, Sequence, Optional, Dict, Any, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

from astra_rl.core.environment import Environment
from astra_rl.core.algorithm import Algorithm
from astra_rl.core.common import ActionT, StateT, Batch, Step
from astra_rl.logging import logger, ASTRAWandbLogger


class ListDataset(Dataset[Step], Generic[Step]):
    def __init__(self, list: Sequence[Step]) -> None:
        super().__init__()
        self.list = list

    def __len__(self) -> int:
        return len(self.list)

    def __getitem__(self, idx: int) -> Step:
        return self.list[idx]


class Harness(Generic[StateT, ActionT, Step, Batch]):
    """Harness for running an algorithm in a given environment.

    Example:

        Here is an example of how to use the `Harness` class with the DPO algorithm
        and an AST problem environment for *one episode only*. You should add your
        own optimization things such as weight decay or scheduling and figure out
        early stopping, etc.

        >>> import torch
        >>> from astra_rl.training.harness import (
        ...     Harness,
        ... )
        >>> from astra_rl.algorithms.dpo import (
        ...     DPO,
        ... )
        >>> from astra_rl.methods.ast import (
        ...     ASTProblem,
        ...     ASTEnvironment,
        ... )
        >>>
        >>> problem = (
        ...     ASTProblem()
        ... )
        >>> environment = (
        ...     ASTEnvironment(
        ...         problem, ...
        ...     )
        ... )
        >>> algorithm = DPO(...)
        >>> harness = Harness(
        ...     environment,
        ...     algorithm,
        ... )
        >>> optimizer = torch.optim.Adam(
        ...     problem.parameters(),
        ...     lr=1e-4,
        ... )
        >>>
        >>> for batch in harness.experience():
        ...     loss = harness.step(
        ...         batch
        ...     )
        ...     loss.backward()
        ...     optimizer.zero_grad()


    Attributes:
        environment (Environment[StateT, ActionT]): The environment to run the algorithm in.
        algorithm (Algorithm[StateT, ActionT, Step, Batch]): The algorithm to run.
        num_episodes_per_experience (int): Number of episodes per call to `.experience()`.
        dataloader_kwargs (Dict[str, Any]): Keyword arguments for the PyTorch data loader. Batch size, for instance, should be set.

    Generics:
        StateT (type): The type of the state in the environment.
        ActionT (type): The type of the action in the environment.
        Step (type): The type of a single step in the environment.
        Batch (type): The type of a batch of steps, passed to the `.step()` function for gradient.
    """

    def __init__(
        self,
        environment: Environment[StateT, ActionT],
        algorithm: Algorithm[StateT, ActionT, Step, Batch],
        num_episodes_per_experience: int = 32,
        use_wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            environment (Environment): The environment to run the algorithm in.
            algorithm (Algorithm): The algorithm to run.
            num_episodes_per_experience (int, optional): Number of episodes per call to `.experience()`. Defaults to 32.
            wandb_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for configuring Weights & Biases. Defaults to None.
            dataloader_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for the PyTorch DataLoader, such as batch size and shuffle. Defaults to None.
        """

        self.environment = environment
        self.algorithm = algorithm
        self.num_episodes_per_experience = num_episodes_per_experience
        self.use_wandb = use_wandb
        self.wandb_kwargs = wandb_kwargs or {}
        self.dataloader_kwargs: Dict[str, Any] = dataloader_kwargs or {}

        if self.use_wandb:
            self.wandb = ASTRAWandbLogger(self.wandb_kwargs)

    def step(self, batch: Batch) -> tuple[torch.Tensor, Dict[Any, Any]]:
        """Run a step of the algorithm on the dataset.

        Args:
            batch (Batch): The dataset batch to run the algorithm on.

        Returns:
            tuple[torch.Tensor, Dict[Any, Any]]: A tuple containing:
                - torch.Tensor: The loss computed by the algorithm (for current batch).
                - Dict[Any, Any]: Additional information for logging.
        """

        result: torch.Tensor
        logging_dict: Dict[Any, Any]
        result, logging_dict = self.algorithm.step(batch)
        step_logs: Dict[Any, Any] = {}

        # TODO: Add other values here to logs besides algorithm specifics? Alternatively, can just return logging_dict
        step_logs = {
            **logging_dict,
        }

        return result, step_logs

    def experience(self, seed: Optional[int] = None) -> Iterator[Batch]:
        """Collect some experiences!

        Args:
            seed (Optional[int], optional): Seed for reproducibility. Defaults to None.

        Returns:
            Sequence[Step]: A sequence of steps collected from the algorithm's rollouts.
        """

        logger.debug(
            f"Collecting {self.num_episodes_per_experience} episodes of experience..."
        )

        graphs = []
        for _ in range(self.num_episodes_per_experience):
            graph = self.environment.rollout(seed=seed)
            graphs.append(graph)
        # for _ in range(self.num_episodes_per_experience):
        #     try:
        #         graph = self.environment.rollout(seed=seed)
        #         graphs.append(graph)
        #     except Exception as e:
        #         print(f"Skipping rollout due to error: {e}")

        steps = sum([list(self.algorithm.flatten(i)) for i in graphs], [])

        logger.debug(
            f"Done collecting {self.num_episodes_per_experience} episodes of experience"
            f", got {len(steps)} training steps."
        )

        return iter(
            DataLoader(
                ListDataset(steps),
                collate_fn=self.algorithm.collate_fn,
                **self.dataloader_kwargs,
            )
        )

    def log_current_step(self, current_logs: Dict[Any, Any]) -> None:
        """Log the current step metrics to Weights & Biases (if enabled) and logger.

        Args:
            current_logs (Dict[Any, Any]): The logs to be recorded.
        """
        if self.use_wandb:
            self.wandb.log(current_logs)

        # Always log to the logger
        # TODO: Do we want to log to the logger? Should be fine as used for debugging?
        logger.info(f"Current logs: {current_logs}")
