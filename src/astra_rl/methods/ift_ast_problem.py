"""
ift_ast_problem.py
AST Problem for Instruction Fine-Tuned Models
"""

from typing import Sequence, Dict

from abc import ABC, abstractmethod
import torch

from astra_rl.core.problem import Problem
from astra_rl.core.moderator import Moderator


class IFTASTProblem(Problem[Sequence[Dict[str, str]], str], ABC):
    """ASTProblem, implemented for instruction fine-tuned models

    For usage examples, see `astra_rl.core.problem.Problem`.

    Attributes:
        moderator (Moderator): The moderator used to evaluate the sequences.
        ppl_weight (float): Weight for perplexity in the reward calculation.
        combined_weight (float): Weight for combined scores in the reward calculation.
        defense_toxicity_weight (float): Weight for defense toxicity in the reward calculation.
    """

    def __init__(
        self,
        moderator: Moderator[Sequence[Dict[str, str]]],
        ppl_weight: float = 0.025,
        combined_weight: float = 0.1,
        defense_toxicity_weight: float = 0.1,
    ):
        super().__init__(moderator)

        self.ppl_weight = ppl_weight
        self.combined_weight = combined_weight
        self.defense_toxicity_weight = defense_toxicity_weight

    def advance(
        self, state: Sequence[Dict[str, str]], action: str, next: str
    ) -> Sequence[Dict[str, str]]:
        return list(state) + [{"role": "assistant", "content": action + next}]

    @abstractmethod
    def rollout_prompt_with_user_model(
        self, x: Sequence[Sequence[Dict[str, str]]]
    ) -> Sequence[Sequence[Dict[str, str]]]:
        """Rolls out a prompt using a model for the user

        Args:
            x (Sequence[StateT]): The input chat messages to roll out from.

        Returns:
            Sequence[StateT]: **REPEAT** the input message + added a user utterance on top.
        """
        pass

    def reward(
        self,
        context: Sequence[Sequence[Dict[str, str]]],
        attack: Sequence[str],
        response: Sequence[str],
    ) -> Sequence[float]:
        # IMPORTANT: this is profound---"combined toxicity" in IFT'd models also includes
        # the context, and defense toxicity is defense only without any context.
        combined_uts = [
            self.advance(i, j, k) for i, j, k in zip(context, attack, response)
        ]
        defender_uts = [self.advance([], "", k) for k in response]
        combined_scores: Sequence[float] = self.moderator.moderate(combined_uts)
        defender_scores: Sequence[float] = self.moderator.moderate(defender_uts)

        # calculate logprobs
        attack_ppls: torch.Tensor = self._get_target_logprobs_and_validate(
            context, attack
        ).exp()

        reward: Sequence[float] = (
            (
                self.ppl_weight * attack_ppls
                + self.combined_weight
                * torch.tensor(combined_scores).to(attack_ppls.device)
                + self.defense_toxicity_weight
                * torch.tensor(defender_scores).to(attack_ppls.device)
            )
            .cpu()
            .tolist()
        )

        return reward
