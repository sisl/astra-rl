"""
environment.py
Roll out a problem, and specify how its environment behaves.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence, Generic, Self, Optional

from astra_rl.core.common import StateT, ActionT
from astra_rl.core.problem import Problem


@dataclass
class Node(Generic[StateT, ActionT]):
    """A node in the rollout graph.

    Represents a single leaf in the rollout process, containing the context,
    the action taken, the response received, the reward for that action,
    and any children nodes that follow this action in this rollout.

    Attributes:
        context (StateT): The initial state before the action.
        attack (ActionT): The action taken in this node.
        response (StateT): The resulting state after the action.
        reward (float): The reward received for taking the action.
        children (Sequence[Node[StateT, ActionT]]): Subsequent nodes that follow this action.

    Generics:
        StateT (type): The type of the state in the environment.
        ActionT (type): The type of the action in the environment.
    """

    context: StateT
    attack: ActionT
    response: StateT
    reward: float

    children: Sequence[Self]


@dataclass
class Graph(Generic[StateT, ActionT]):
    """A graph representing the rollout (history + actions) of a problem.

    Attributes:
        context (StateT): The initial state of the environment.
        children (Sequence[Node[StateT, ActionT]]): The sequence of nodes representing actions and responses.
    """

    context: StateT
    children: Sequence[Node[StateT, ActionT]]


class Environment(ABC, Generic[StateT, ActionT]):
    """An Environment used for rolling out a problem.

    The primary point of this class is to make a `Graph` of the problem
    by calling the `rollout` method. The environment can keep/sample
    initial state, but should not have global state that persists
    across rollouts.

    Attributes:
        problem (Problem[StateT, ActionT]): The problem instance that defines the environment and actions.

    Generics:
        StateT (type): The type of the state in the environment.
        ActionT (type): The type of the action in the environment.
    """

    def __init__(self, problem: Problem[StateT, ActionT]):
        self.problem = problem

    @abstractmethod
    def rollout(self, seed: Optional[int] = None) -> Graph[StateT, ActionT]:
        """Roll out a problem and return a graph of the actions taken.

        Args:
            seed (Optional[int]): An optional seed; the same seed should produce the same graph.

        Returns:
            Graph[StateT, ActionT]: A graph representing the rollout of the problem.
        """

        pass

    @abstractmethod
    def eval_rollout(self, prompt: StateT) -> Graph[StateT, ActionT]:
        """Roll out a problem following a SPECIFIC state for evaulations.

        Args:
            prompt (StateT): The state to begin to roll out from.

        Returns:
            Graph[StateT, ActionT]: A graph representing the rollout of the problem.
        """

        pass
