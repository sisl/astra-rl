from dataclasses import dataclass
from typing import Generic, Sequence, List, Any, Dict

from astra_rl.core.algorithm import Algorithm
from astra_rl.core.system import System
from astra_rl.core.common import StateT, ActionT
from astra_rl.core.sampler import Graph

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
    """Direct Preference Optimization (DPO) algorithm."""

    def __init__(self, system: System[StateT, ActionT], beta: float = 0.1):
        super().__init__(system)

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
                    suffix_pos=pos_entry.probe,
                    suffix_neg=neg_entry.probe,
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

    def step(
        self, batch: DPOBatch[StateT, ActionT]
    ) -> tuple[torch.Tensor, Dict[Any, Any]]:
        tester_logprobs_win = self.system._get_tester_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        ).sum(dim=-1)  # Sum per-token logprobs to get sequence logprobs
        tester_logprobs_loss = self.system._get_tester_logprobs_and_validate(
            batch.prefixes, batch.suffix_neg
        ).sum(dim=-1)  # Sum per-token logprobs to get sequence logprobs
        baseline_logprobs_win = self.system._get_baseline_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        ).sum(dim=-1)  # Sum per-token logprobs to get sequence logprobs
        baseline_logprobs_loss = self.system._get_baseline_logprobs_and_validate(
            batch.prefixes, batch.suffix_neg
        ).sum(dim=-1)  # Sum per-token logprobs to get sequence logprobs

        # https://github.com/eric-mitchell/direct-preference-optimization/blob/ \
        # f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L70-L87
        pi_logratios = tester_logprobs_win - tester_logprobs_loss
        ref_logratios = baseline_logprobs_win - baseline_logprobs_loss
        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(self.beta * logits)

        # Calculate addition quantities
        # TODO: CHECK ME for correctness and completion!
        chosen_rewards = self.beta * (tester_logprobs_win - baseline_logprobs_win)
        rejected_rewards = self.beta * (tester_logprobs_loss - baseline_logprobs_loss)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margin = chosen_rewards - rejected_rewards

        logging_dict: Dict[Any, Any] = {
            "training/loss": loss.mean().cpu().item(),
            "reward/chosen_rewards": chosen_rewards.mean().cpu().item(),
            "reward/rejected_rewards": rejected_rewards.mean().cpu().item(),
            "reward/reward_accuracies": reward_accuracies.mean().cpu().item(),
            "reward/reward_margin": reward_margin.mean().cpu().item(),
            "policy/logprobs_chosen": tester_logprobs_win.mean().detach().cpu().item(),
            "policy/logprobs_rejected": tester_logprobs_loss.mean()
            .detach()
            .cpu()
            .item(),
            "ref/logprobs_chosen": baseline_logprobs_win.mean().detach().cpu().item(),
            "ref/logprobs_rejected": baseline_logprobs_loss.mean()
            .detach()
            .cpu()
            .item(),
        }
        # TODO: Add this from old code?
        # "policy/rollout": wandb.Html(str(r"<span>"+batch["prompt_win"][0][0]+"</span><span style='color:Tomato;'>"+batch["prompt_win"][0][1]+r"</span><span style='color:DodgerBlue'>"+batch["prompt_win"][0][2]+r"</span>")),

        return loss.mean(), logging_dict


class IPO(DPO[StateT, ActionT]):
    def step(
        self, batch: DPOBatch[StateT, ActionT]
    ) -> tuple[torch.Tensor, Dict[Any, Any]]:
        tester_logprobs_win = self.system._get_tester_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        ).sum(dim=-1)  # Sum per-token logprobs to get sequence logprobs
        tester_logprobs_loss = self.system._get_tester_logprobs_and_validate(
            batch.prefixes, batch.suffix_neg
        ).sum(dim=-1)  # Sum per-token logprobs to get sequence logprobs
        baseline_logprobs_win = self.system._get_baseline_logprobs_and_validate(
            batch.prefixes, batch.suffix_pos
        ).sum(dim=-1)  # Sum per-token logprobs to get sequence logprobs
        baseline_logprobs_loss = self.system._get_baseline_logprobs_and_validate(
            batch.prefixes, batch.suffix_neg
        ).sum(dim=-1)  # Sum per-token logprobs to get sequence logprobs

        # https://github.com/eric-mitchell/direct-preference-optimization/blob/ \
        # f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L70-L87
        pi_logratios = tester_logprobs_win - tester_logprobs_loss
        ref_logratios = baseline_logprobs_win - baseline_logprobs_loss
        logits = pi_logratios - ref_logratios

        loss = (logits - 1 / (2 * self.beta)) ** 2

        # Calculate addition quantities
        # TODO: CHECK ME for correctness and completion!
        chosen_rewards = self.beta * (tester_logprobs_win - baseline_logprobs_win)
        rejected_rewards = self.beta * (tester_logprobs_loss - baseline_logprobs_loss)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margin = chosen_rewards - rejected_rewards

        logging_dict: Dict[Any, Any] = {
            "training/loss": loss.mean().cpu().item(),
            "reward/chosen_rewards": chosen_rewards.mean().cpu().item(),
            "reward/rejected_rewards": rejected_rewards.mean().cpu().item(),
            "reward/reward_accuracies": reward_accuracies.mean().cpu().item(),
            "reward/reward_margin": reward_margin.mean().cpu().item(),
            "policy/logprobs_chosen": tester_logprobs_win.mean().detach().cpu().item(),
            "policy/logprobs_rejected": tester_logprobs_loss.mean()
            .detach()
            .cpu()
            .item(),
            "ref/logprobs_chosen": baseline_logprobs_win.mean().detach().cpu().item(),
            "ref/logprobs_rejected": baseline_logprobs_loss.mean()
            .detach()
            .cpu()
            .item(),
        }
        # TODO: Add this from old code?
        # "policy/rollout": wandb.Html(str(r"<span>"+batch["prompt_win"][0][0]+"</span><span style='color:Tomato;'>"+batch["prompt_win"][0][1]+r"</span><span style='color:DodgerBlue'>"+batch["prompt_win"][0][2]+r"</span>")),

        return loss.mean(), logging_dict
