"""
moderator.py
"""

from abc import ABC, abstractmethod
from typing import Generic, Union, Sequence

from astra_rl.core.common import StateT


class Moderator(ABC, Generic[StateT]):
    """Red-Teaming moderator for evaluating sequences."""

    @abstractmethod
    def moderate(self, x: Sequence[Union[StateT]]) -> Sequence[float]:
        pass
