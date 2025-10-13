"""
scorer.py - Scoring system for evaluating sequences and nodes.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Union, Sequence, Dict, Any, TYPE_CHECKING, Callable

from astra_rl.core.common import StateT, ActionT

if TYPE_CHECKING:
    from astra_rl.core.sampler import Node


class ScoringMode(Enum):
    """Modes for when/how to apply scoring.

    TURN: Score each turn independently (challenge and response separately).
    CUMULATIVE: Score with full conversation history up to this point.
    FINAL: Only score complete conversations (leaf nodes only).
    """

    TURN = "turn"
    CUMULATIVE = "cumulative"
    FINAL = "final"


class Scorer(ABC, Generic[StateT, ActionT]):
    """Base scorer for evaluating sequences and nodes.

    Scorers can operate in different modes (TURN, CUMULATIVE, FINAL) to control
    when and how scoring is applied during rollouts.

    Attributes:
        mode: The scoring mode for this scorer.
    """

    def __init__(self, mode: ScoringMode = ScoringMode.TURN):
        self.mode = mode

    @abstractmethod
    def score(self, x: Sequence[Union[StateT, ActionT]]) -> Sequence[float]:
        """Score sequences (backward compatibility method).

        Args:
            x: Sequence of items to score.

        Returns:
            Sequence of scores, one per input item.
        """
        pass

    def score_node(self, node: "Node[StateT, ActionT]") -> Dict[str, Any]:
        """Score a node based on the scorer's mode.

        This method should be overridden by subclasses to provide mode-specific
        scoring logic. The default implementation uses the basic score() method.

        Args:
            node: The node to score.

        Returns:
            Dictionary of score names to score values.
        """
        # Default implementation - subclasses should override for mode-specific logic
        if self.mode == ScoringMode.TURN:
            # Score challenge and response separately
            scores = {}
            if node.challenge:
                challenge_scores = self.score([node.challenge])
                scores["challenge_score"] = challenge_scores[0]
            response_scores = self.score([node.response])
            scores["response_score"] = response_scores[0]
            return scores
        elif self.mode == ScoringMode.CUMULATIVE:
            # Score full conversation up to this point
            full_text = node.get_conversation_text()
            score_value = self.score([full_text])[0]
            return {
                "cumulative_score": score_value,
                "depth": node.depth,
            }
        else:  # FINAL
            # Score complete conversation (only meaningful on leaf nodes)
            if node.is_leaf():
                full_text = node.get_conversation_text()
                score_value = self.score([full_text])[0]
                return {"final_score": score_value}
            return {}


class LegacyScorer(Scorer[StateT, ActionT]):
    """Wrapper for legacy scorers that don't support modes.

    This allows old scorers that only implement score(x) to work with
    the new system without modification.
    """

    def __init__(
        self,
        score_fn: Callable[[Sequence[Union[StateT, ActionT]]], Sequence[float]],
    ):
        super().__init__(mode=ScoringMode.TURN)
        self._score_fn = score_fn

    def score(self, x: Sequence[Union[StateT, ActionT]]) -> Sequence[float]:
        return list(self._score_fn(x))

    def score_node(self, node: "Node[StateT, ActionT]") -> Dict[str, Any]:
        # Use default TURN mode implementation
        return super().score_node(node)
