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
    def __init__(
        self,
        environment: Environment[StateT, ActionT],
        algorithm: Algorithm[StateT, ActionT, Step, Batch],
        num_episodes_per_experience: int = 32,
        **kwargs: Any,
    ) -> None:
        """Harness for running an algorithm in a given environment.

        Args:
            environment (Environment[StateT, ActionT]): The environment to run the algorithm in.
            algorithm (Algorithm[StateT, ActionT, Step, Batch]): The algorithm to run.
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
