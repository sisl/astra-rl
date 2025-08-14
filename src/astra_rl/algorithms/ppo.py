from dataclasses import dataclass
from abc import ABC
from typing import Generic, Sequence, List, Any, Dict

from astra_rl.core.algorithm import Algorithm
from astra_rl.core.problem import ValueFunctionProblem
from astra_rl.core.common import StateT, ActionT
from astra_rl.core.environment import Graph

import torch
import torch.nn.functional as F


@dataclass
class PPOStep(Generic[StateT, ActionT]):
    prefix: StateT
    suffix: ActionT
    reward: float


@dataclass
class PPOBatch(Generic[StateT, ActionT]):
    prefix: Sequence[StateT]
    suffix: Sequence[ActionT]
    reward: Sequence[float]


class PPO(
    Algorithm[StateT, ActionT, PPOStep[StateT, ActionT], PPOBatch[StateT, ActionT]],
    ABC,
):
    """Proximal Policy Optimization (PPO) algorithm with value function."""

    def __init__(
        self,
        problem: ValueFunctionProblem[StateT, ActionT],
        clip_range: float = 0.1,
        vf_loss_coef: float = 1.0,
    ):
        super().__init__(problem)

        self.problem: ValueFunctionProblem[StateT, ActionT] = problem
        self.clip_range = clip_range
        self.vf_loss_coef = vf_loss_coef

    def flatten(
        self, graph: Graph[StateT, ActionT]
    ) -> Sequence[PPOStep[StateT, ActionT]]:
        # in DPO, we sample from each branch the most rewarded
        # and least rewarded actions in order to use them as our contrastive
        # pairs.

        res: List[PPOStep[StateT, ActionT]] = []
        bfs = [graph.children]
        while len(bfs):
            front = bfs.pop(0)
            if len(list(front)) < 2:
                # if there is no pair, we skip this node
                continue

            for i in front:
                res.append(PPOStep(prefix=i.context, suffix=i.attack, reward=i.reward))
                bfs.append(i.children)

        return res

    @staticmethod
    def collate_fn(x: Sequence[PPOStep[StateT, ActionT]]) -> PPOBatch[StateT, ActionT]:
        prefixes = [i.prefix for i in x]
        suffix = [i.suffix for i in x]
        rewards = [i.reward for i in x]

        return PPOBatch(prefix=prefixes, suffix=suffix, reward=rewards)

    def step(
        self, batch: PPOBatch[StateT, ActionT]
    ) -> tuple[torch.Tensor, Dict[Any, Any]]:
        logprobs_attacker = self.problem._get_attacker_logprobs_and_validate(
            batch.prefix, batch.suffix
        )
        logprobs_baseline = self.problem._get_baseline_logprobs_and_validate(
            batch.prefix, batch.suffix
        )
        values = self.problem.value(batch.prefix, batch.suffix)

        # Q(s,a) = R(s,a), which is jank but seems to be the standard
        # also its bootstrapped without discount throughout the stream
        Q = (
            torch.tensor(batch.reward)
            .to(logprobs_attacker.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, *values.shape[1:])
        )
        A = Q - values

        # normalize advantages
        if A.size(-1) == 1:
            A = ((A - A.mean()) / (A.std() + 1e-8)).squeeze(-1)
        else:
            A = (A - A.mean()) / (A.std() + 1e-8)
        # compute ratio, should be 1 at the first iteration
        ratio = torch.exp((logprobs_attacker - logprobs_baseline.detach()))

        # compute clipped surrogate lolss
        policy_loss_1 = A * ratio
        policy_loss_2 = A * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss_2 = A * torch.clamp(ratio, 1 - 0.1, 1 + 0.1)
        policy_loss = -(torch.min(policy_loss_1, policy_loss_2)).mean()

        # compute value loss
        value_loss = F.mse_loss(Q, values)

        # compute final lossvalue_loss
        loss = policy_loss + self.vf_loss_coef * value_loss

        # create logging dict
        logging_dict: Dict[Any, Any] = {
            "training/loss": loss.mean().cpu().item(),
            "training/policy_loss": policy_loss.mean().cpu().item(),
            "training/value_loss": value_loss.mean().cpu().item(),
            "reward/mean_reward": torch.tensor(batch.reward).mean().cpu().item(),
            "reward/std_reward": torch.tensor(batch.reward).std().cpu().item(),
            "policy/logprobs": logprobs_attacker.mean().detach().cpu().item(),
            "ref/logprobs": logprobs_baseline.mean().detach().cpu().item(),
        }

        return loss, logging_dict
