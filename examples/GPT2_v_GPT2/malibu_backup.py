"""
malibu.py
This uses the ast environment and Huggingface extensions. However, with modifications to the environment,
we can enable the backup of discounted future rewards during multi-turn conversation rollouts used in PPO.

This example shows how to use the Astra RL library to train a GPT-2 model
using the DPO algorithm with a Detoxify moderator for content moderation.

We will train using the ConvoKit Reddit Small Corpus for initial prompts, filtered by the moderator to be non-toxic.
"""

import json
from astra_rl.ext.transformers.hf_ast_problem import (
    HFASTTrainer,
    HFASTConfiguration,
)

from astra_rl import ASTEnvironment, DPO, ASTNode

# GPT2 v GPT2 with Detoxify Moderator
from ast_basic_1 import ExampleDetoxifyProblem

# GPT2 v GPT2 with LlamaGuard Moderator
from astra_rl.core.environment import Graph


class MalibuEnv(ASTEnvironment):
    def __init__(self, problem, prompts, gamma=0.9, tree_width=2, tree_depth=3):
        super().__init__(problem, prompts, tree_width, tree_depth)
        self.gamma = gamma

    # leave handle_prompt and rollout alone, after rollout we will backup rewards
    def backup_rewards(self, graph: "Graph[str, str]") -> "Graph[str, str]":
        """
        In-place backup of rewards up the rollout tree.

        For each node:
            node.reward <- node.reward + gamma * max(child.reward_after_backup)

        Leaves keep their original reward. The Graph root has no reward field,
        so we start from its children and back up to the top.

        Returns the same graph object with updated rewards.
        """

        gamma = float(self.gamma)

        def _backup_node(n: "ASTNode") -> float:
            # Base case: leaf node, nothing to back up
            if not n.children:
                return float(n.reward)

            # Recursively back up children first
            best_child_value = max(_backup_node(c) for c in n.children)

            # Update this node's reward using the winning child's (max) value
            n.reward = float(n.reward) + gamma * best_child_value
            return float(n.reward)

        # Graph has no reward; back up each top-level child
        for child in graph.children:
            _backup_node(child)

        return graph


# main code body
def main() -> None:
    # read in training prompts
    with open("prompts_reddit_train.json") as f:
        PROMPTS = json.load(f)

    # read in dev set of prompts
    with open("prompts_reddit_dev.json") as f:
        DEV_PROMPTS = json.load(f)

    DEVICE = "cuda"  # cuda/cpu/mps

    # instatiate our problem and environment
    # instatiate our problem with GPT-2 attacker, target, and baseline
    # TODO HFASTProblem needs to be updated to ensure device-side out of indexing error (same issue as earlier Detoxify Problem)
    # bend can't handle this much memory
    # problem = HFASTProblem("meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B", DetoxifyModerator(), DEVICE)

    problem = ExampleDetoxifyProblem(DEVICE)
    env = MalibuEnv(problem, PROMPTS, gamma=0.9)

    # # print out rollouts - checked that backing up rewards works!
    # for i in range(1):
    #     rollout = env.rollout()
    #     print(f"ROLLOUT {i}")
    #     print(rollout)
    #     print("\n\n")

    #     new_rollout = env.backup_rewards(rollout)
    #     print(f"ROLLOUT {i} after reward backup")
    #     print(new_rollout)
    #     print("\n\n")

    # no changes from here on out!
    solver = DPO(problem)
    # instantiate the pre-configured HF-compatable traininer class
    config = HFASTConfiguration()
    trainer = HFASTTrainer(
        config,
        env,
        solver,
        dev_prompts=DEV_PROMPTS,
        eval_every=100,
        ckpt_dir="checkpoints",
    )
    trainer.train()


if __name__ == "__main__":
    main()
