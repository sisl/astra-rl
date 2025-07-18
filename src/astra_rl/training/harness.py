from typing import Generic, Sequence, Optional, Dict, Any, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

from astra_rl.core.environment import Environment
from astra_rl.core.algorithm import Algorithm
from astra_rl.core.common import ActionT, StateT, Batch, Step
from astra_rl.logging import logger


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
        **kwargs: Any,
    ) -> None:
        """
        Args:
            environment (Environment): The environment to run the algorithm in.
            algorithm (Algorithm): The algorithm to run.
            rollouts_per_eps (int, optional): Number of episodes per call to `.experience()`. Defaults to 32.
            **kwargs: keyword arguments for the PyTorch data loader constructed on the fly
        """

        self.environment = environment
        self.algorithm = algorithm
        self.num_episodes_per_experience = num_episodes_per_experience
        self.dataloader_kwargs: Dict[str, Any] = kwargs

    def step(self, batch: Batch) -> torch.Tensor:
        """Run a step of the algorithm on the dataset.

        Args:
            dataset (ListDataset): The dataset to run the algorithm on.

        Returns:
            torch.Tensor: The loss computed by the algorithm.
        """

        result: torch.Tensor = self.algorithm.step(batch)
        return result

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
