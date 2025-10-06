"""
sampler.py
Roll out a system, and specify how its sampler behaves.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence, Generic, Self, Optional, Any

from astra_rl.core.common import StateT, ActionT
from astra_rl.core.system import System


@dataclass
class Node(Generic[StateT, ActionT]):
    """A node in the rollout graph.

    Represents a single leaf in the rollout process, containing the context,
    the action taken, the response received, the reward for that action,
    and any children nodes that follow this action in this rollout.

    Attributes:
        context (StateT): The initial state before the action.
        utterance (ActionT): The action taken in this node.
        response (StateT): The resulting state after the action.
        reward (float): The reward received for taking the action.
        children (Sequence[Node[StateT, ActionT]]): Subsequent nodes that follow this action.

    Generics:
        StateT (type): The type of the state in the sampler.
        ActionT (type): The type of the action in the sampler.
    """

    context: StateT
    utterance: ActionT
    response: StateT
    reward: float

    children: Sequence[Self]


@dataclass
class Graph(Generic[StateT, ActionT]):
    """A graph representing the rollout (history + actions) of a system.

    Attributes:
        context (StateT): The initial state of the sampler.
        children (Sequence[Node[StateT, ActionT]]): The sequence of nodes representing actions and responses.
    """

    context: StateT
    children: Sequence[Node[StateT, ActionT]]


class Sampler(ABC, Generic[StateT, ActionT]):
    """A Sampler used for rolling out a system.

    The primary point of this class is to make a `Graph` of the system
    by calling the `rollout` method. The sampler can keep/sample
    initial state, but should not have global state that persists
    across rollouts.

    Attributes:
        system (System[StateT, ActionT]): The system instance that defines the sampler and actions.

    Generics:
        StateT (type): The type of the state in the sampler.
        ActionT (type): The type of the action in the sampler.
    """

    def __init__(self, system: System[StateT, ActionT]):
        self.system = system

    @abstractmethod
    def rollout(self, seed: Optional[int] = None) -> Graph[StateT, ActionT]:
        """Roll out a system and return a graph of the actions taken.

        Args:
            seed (Optional[int]): An optional seed; the same seed should produce the same graph.

        Returns:
            Graph[StateT, ActionT]: A graph representing the rollout of the system.
        """

        pass

    def eval_rollout(self, seed: Optional[Any] = None) -> Graph[StateT, ActionT]:
        """Roll out for evaluation, by default just the standard rollout

        Notes:
            This can be customized to whatever the user desires in terms of rollout for eval.
            For instance, for evaluation the seed maybe StateT instead of int since there may
            be another evaluation dataset.

            However, if the seed given is None or an int, a default implementation exists
            which just calls `self.rollout(seed)` and so evaluation can be done without
            needing to override this method.

        Args:
            seed (Optional[Any]): An optional seed; the same seed should produce the same graph.

        Returns:
            Graph[StateT, ActionT]: A graph representing the rollout of the system.
        """

        if seed is None or isinstance(seed, int):
            return self.rollout(seed)

        raise NotImplementedError(
            "eval_rollout not implemented for non-int seeds; please override this method."
        )
