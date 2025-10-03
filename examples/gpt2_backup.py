"""
backup.py
This uses the ast sampler and Huggingface extensions. However, with modifications to the sampler,
we can enable the backup of discounted future rewards during multi-turn conversation rollouts used in PPO.

This example shows how to use the Astra RL library to train a GPT-2 model
using the DPO algorithm with a Detoxify scorer for content moderation.

We will train using the ConvoKit Reddit Small Corpus for initial prompts, filtered by the scorer to be non-toxic.
"""

from astra_rl.datasets import CONVOKIT_REDDIT_TRAIN, CONVOKIT_REDDIT_DEV
from astra_rl.ext.transformers.hf_ast_system import (
    HFASTTrainer,
    HFASTConfiguration,
)

from astra_rl import DPO, ASTNode, ASTSampler

# GPT2 v GPT2 with Detoxify Scorer
from ast_gpt2 import GPT2DetoxifySystem

# GPT2 v GPT2 with LlamaGuard Scorer
from astra_rl.core.sampler import Graph


class BackupSampler(ASTSampler):
    def __init__(self, system, prompts, gamma=0.9, tree_width=2, tree_depth=3):
        super().__init__(system, prompts, tree_width, tree_depth)
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
    DEVICE = "cuda"  # cuda/cpu/mps

    # instatiate our system and sampler
    # instatiate our system with GPT-2 tester, target, and baseline

    # bend can't handle this much memory
    # system = HFASTSystem("meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B", DetoxifyScorer(), DEVICE)

    system = GPT2DetoxifySystem(DEVICE)
    sampler = BackupSampler(system, CONVOKIT_REDDIT_TRAIN, gamma=0.9)

    # # print out rollouts - checked that backing up rewards works!
    # for i in range(1):
    #     rollout = sampler.rollout()
    #     print(f"ROLLOUT {i}")
    #     print(rollout)
    #     print("\n\n")

    #     new_rollout = sampler.backup_rewards(rollout)
    #     print(f"ROLLOUT {i} after reward backup")
    #     print(new_rollout)
    #     print("\n\n")

    # no changes from here on out!
    solver = DPO(system)
    # instantiate the pre-configured HF-compatable traininer class
    config = HFASTConfiguration()
    trainer = HFASTTrainer(
        config,
        sampler,
        solver,
        dev_prompts=CONVOKIT_REDDIT_DEV,
        eval_every=100,
        ckpt_dir="checkpoints",
    )
    trainer.train()


if __name__ == "__main__":
    main()
