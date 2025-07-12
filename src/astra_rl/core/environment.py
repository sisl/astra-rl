"""
rollout.py
Roll out a problem
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence, Generic, Self, Optional

from astra_rl.core.common import StateT, ActionT
from astra_rl.core.problem import Problem


@dataclass
class Node(Generic[StateT, ActionT]):
    context: StateT
    attack: ActionT
    response: StateT
    reward: float

    children: Sequence[Self]


@dataclass
class Graph(Generic[StateT, ActionT]):
    context: StateT
    children: Sequence[Node[StateT, ActionT]]


class Environment(ABC, Generic[StateT, ActionT]):
    def __init__(self, problem: Problem[StateT, ActionT]):
        self.problem = problem

    @abstractmethod
    def rollout(self, seed: Optional[int] = None) -> Graph[StateT, ActionT]:
        pass
