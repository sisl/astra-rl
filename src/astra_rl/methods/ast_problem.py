"""
ast_problem.py
ASTProblem
"""

import random
from random import Random
from types import ModuleType
from typing import Sequence, Optional, Union

import torch

from astra_rl.core.problem import Problem
from astra_rl.core.moderator import Moderator
from astra_rl.core.environment import Node, Graph, Environment


class ASTProblem(Problem[str, str]):
    """Adaptive Stress Testing for Large Language Model Red-Teaming.

    Implements https://arxiv.org/abs/2407.09447.

    For usage examples, see `astra_rl.core.problem.Problem`.

    Attributes:
        moderator (Moderator): The moderator used to evaluate the sequences.
        ppl_weight (float): Weight for perplexity in the reward calculation.
        combined_weight (float): Weight for combined scores in the reward calculation.
        defense_toxicity_weight (float): Weight for defense toxicity in the reward calculation.
    """

    def __init__(
        self,
        moderator: Moderator[str, str],
        ppl_weight: float = 0.025,
        combined_weight: float = 0.1,
        defense_toxicity_weight: float = 0.1,
    ):
        super().__init__(moderator)

        self.ppl_weight = ppl_weight
        self.combined_weight = combined_weight
        self.defense_toxicity_weight = defense_toxicity_weight

    def advance(self, state: str, action: str, next: str) -> str:
        return state + action + next

    def reward(
        self, context: Sequence[str], attack: Sequence[str], response: Sequence[str]
    ) -> Sequence[float]:
        combined_uts = [j + k for j, k in zip(attack, response)]
        combined_scores: Sequence[float] = self.moderator.moderate(combined_uts)
        defender_scores: Sequence[float] = self.moderator.moderate(response)
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


ASTNode = Node[str, str]


class ASTEnvironment(Environment[str, str]):
    """The ASTPrompter Rollout Environment

    Implements https://arxiv.org/abs/2407.09447.

    Specifically, this is the original rollout system used in the
    ASTPrompter paper, the case of red-teaming where we have
    the attacker and defender generates successive turns of strings,
    each of which is appended to the prompt of the other. They do not
    have IFT or other types of structure.

    For usage examples, see `astra_rl.core.environment.Environment`.

    Attributes:
        problem (ASTProblem): The problem instance that defines the environment and actions.
        prompts (Sequence[str]): A sequence of initial prompts to start the rollout.
        tree_width (int): The number of branches at each node in the rollout tree.
        tree_depth (int): The depth of the rollout tree.

    Generics:
        StateT (str): The type of the state in the environment, which is a string.
        ActionT (str): The type of the action in the environment, which is also a string.
    """

    def __init__(
        self,
        problem: ASTProblem,
        prompts: Sequence[str],
        tree_width: int = 2,
        tree_depth: int = 3,
    ):
        super().__init__(problem)

        self.prompts = prompts
        self.tree_width = tree_width
        self.tree_depth = tree_depth

    def __handle_prompt(self, prompt: str, depth: int = 3) -> Sequence[Node[str, str]]:
        if depth == 0:
            return []

        prompts = [prompt for _ in range(self.tree_width)]
        attacks = self.problem._rollout_prompt_with_attacker_and_validate(prompts)
        defenses = self.problem._rollout_prompt_with_target_and_validate(
            [prompt + i for i in attacks]
        )
        rewards = self.problem.reward(prompts, attacks, defenses)

        nodes = [
            Node(
                prompt,
                attack,
                defense,
                reward,
                self.__handle_prompt(
                    self.problem.advance(prompt, attack, defense), depth - 1
                ),
            )
            for prompt, attack, defense, reward in zip(
                prompts, attacks, defenses, rewards
            )
        ]

        return nodes

    def rollout(self, seed: Optional[int] = None) -> Graph[str, str]:
        R: Union[Random, ModuleType] = random
        if seed:
            R = Random(seed)
        prompt = R.choice(self.prompts)
        nodes = self.__handle_prompt(prompt, self.tree_depth)

        return Graph(prompt, nodes)
