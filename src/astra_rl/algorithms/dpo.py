from dataclasses import dataclass
from typing import Generic, Sequence, List

from astra_rl.core.algorithm import Algorithm
from astra_rl.core.problem import Problem
from astra_rl.core.common import StateT, ActionT
from astra_rl.core.environment import Graph

import torch
import torch.nn.functional as F


@dataclass
class DPOStep(Generic[StateT, ActionT]):
    prefix: StateT

    suffix_pos: ActionT
    suffix_neg: ActionT


@dataclass
class DPOBatch(Generic[StateT, ActionT]):
    prefixes: Sequence[StateT]

    suffix_pos: Sequence[ActionT]
    suffix_neg: Sequence[ActionT]


class DPO(
    Algorithm[StateT, ActionT, DPOStep[StateT, ActionT], DPOBatch[StateT, ActionT]],
    Generic[StateT, ActionT],
):
    def __init__(self, problem: Problem[StateT, ActionT], beta: float = 0.1):
        super().__init__(problem)

        self.beta = beta

    def flatten(
        self, graph: Graph[StateT, ActionT]
    ) -> Sequence[DPOStep[StateT, ActionT]]:
        # in DPO, we sample from each branch the most rewarded
        # and least rewarded actions in order to use them as our contrastive
        # pairs.

        pairs: List[DPOStep[StateT, ActionT]] = []
        bfs = [graph.children]
        while len(bfs):
            front = bfs.pop(0)
            sorted_list = sorted(list(front), key=lambda x: x.reward, reverse=True)

            if len(sorted_list) < 2:
                # if there is no pair, we skip this node
                continue

            pos_entry = sorted_list[0]
            neg_entry = sorted_list[-1]

            assert pos_entry.context == neg_entry.context, (
                "paired rollouts for DPO must share the same prefix!"
            )

            pairs.append(
                DPOStep(
                    prefix=pos_entry.context,
                    suffix_pos=pos_entry.attack,
                    suffix_neg=neg_entry.attack,
                )
            )

            for i in sorted_list:
                bfs.append(i.children)

        return pairs

    @staticmethod
    def collate_fn(x: Sequence[DPOStep[StateT, ActionT]]) -> DPOBatch[StateT, ActionT]:
        prefixes = [i.prefix for i in x]
        suffix_pos = [i.suffix_pos for i in x]
        suffix_neg = [i.suffix_neg for i in x]

        return DPOBatch(prefixes=prefixes, suffix_pos=suffix_pos, suffix_neg=suffix_neg)

    def step(self, batch: DPOBatch[StateT, ActionT]) -> torch.Tensor:
        attacker_logprobs_win = self.problem._get_attacker_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        )
        attacker_logprobs_loss = self.problem._get_attacker_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        )
        baseline_logprobs_win = self.problem._get_baseline_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        )
        baseline_logprobs_loss = self.problem._get_baseline_logprobs_and_validate(
            batch.prefixes, batch.suffix_neg
        )

        # https://github.com/eric-mitchell/direct-preference-optimization/blob/ \
        # f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L70-L87
        pi_logratios = attacker_logprobs_win - attacker_logprobs_loss
        ref_logratios = baseline_logprobs_win - baseline_logprobs_loss
        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(self.beta * logits)

        # TODO! how do we do logging?
        # ideally there's a logging package / logger for metrics that I can just log to
        # chosen_rewards = self.beta * (attacker_logprob_win - reference_logprobs_win).detach()
        # rejected_rewards = self.beta * (attacker_logprob_loose - referenge_logprobs_loose).detach()

        return loss.mean()


class IPO(DPO[StateT, ActionT]):
    def step(self, batch: DPOBatch[StateT, ActionT]) -> torch.Tensor:
        attacker_logprobs_win = self.problem._get_attacker_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        )
        attacker_logprobs_loss = self.problem._get_attacker_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        )
        baseline_logprobs_win = self.problem._get_baseline_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        )
        baseline_logprobs_loss = self.problem._get_baseline_logprobs_and_validate(
            batch.prefixes, batch.suffix_neg
        )

        # https://github.com/eric-mitchell/direct-preference-optimization/blob/ \
        # f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L70-L87
        pi_logratios = attacker_logprobs_win - attacker_logprobs_loss
        ref_logratios = baseline_logprobs_win - baseline_logprobs_loss
        logits = pi_logratios - ref_logratios

        loss = (logits - 1 / (2 * self.beta)) ** 2

        return loss.mean()
