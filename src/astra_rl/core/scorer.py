"""
scorer.py
"""

from abc import ABC, abstractmethod
from typing import Generic, Union, Sequence

from astra_rl.core.common import StateT, ActionT


class Scorer(ABC, Generic[StateT, ActionT]):
    """Red-Teaming scorer for evaluating sequences."""

    @abstractmethod
    def score(self, x: Sequence[Union[StateT, ActionT]]) -> Sequence[float]:
        pass
