"""
algorithm.py
"""

from abc import abstractmethod, ABC
from typing import Sequence, Generic, Any

import torch

from astra_rl.core.problem import Problem
from astra_rl.core.environment import Graph
from astra_rl.core.common import Step, Batch, StateT, ActionT


class Algorithm(ABC, Generic[StateT, ActionT, Step, Batch]):
    """An Algorithm used for performing training.

    Specifically, the Algorithm object is responsible for encoding
    how a particular rollout graph becomes processed into a loss
    which updates the weights of the model. To implement its children,
    you basically call self.problem's various methods to push values
    through the network.


    Attributes:
        problem (Problem): The problem instance that defines the environment and actions.

    Generics:
        StateT (type): The type of the state in the environment.
        ActionT (type): The type of the action in the environment.
        Step (type): The type of a single step in the environment.
        Batch (type): The type of a batch of steps, passed to the .step() function for gradient.
    """

    def __init__(self, problem: Problem[StateT, ActionT]):
        self.problem = problem

    @abstractmethod
    def flatten(self, graph: Graph[StateT, ActionT]) -> Sequence[Step]:
        """Process a rollout graph into a sequence of steps.

        Args:
            graph (Graph[StateT, ActionT]): The graph to flatten.

        Returns:
            Sequence[Step]: A sequence of steps representing the flattened graph.
        """
        pass

    @staticmethod
    @abstractmethod
    def collate_fn(batch: Sequence[Step]) -> Batch:
        """The collate_fn for torch dataloaders for batching.

        We use this as the literal collate_fn to a torch DataLoader, and
        it is responsible for emitting well-formed batches of data.

        Args:
            batch (Sequence[Step]): A sequence of steps to collate.

        Returns:
            Batch: A batch of data ready for processing using .step().
        """
        pass

    @abstractmethod
    def step(self, batch: Batch) -> tuple[torch.Tensor, dict[Any, Any]]:
        """Take a batch and compute loss of this batch.

        Args:
            batch (Batch): A batch of data to process.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        pass
