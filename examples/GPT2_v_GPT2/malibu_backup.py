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
    HFASTProblem,
)

from astra_rl import ASTEnvironment, DetoxifyModerator


# malibu magic time! Create a custom environment
class MalibuEnv(ASTEnvironment):
    def __init__(self, problem, prompts, gamma=0.9, tree_width=2, tree_depth=3):
        super().__init__(problem, prompts, tree_width, tree_depth)
        self.gamma = gamma

    # leave handle_prompt and rollout alone, after rollout we will backup rewards
    def backup_rewards(self, graph):
        # start at leaf nodes, backup winning child node to parent with gamma discount
        pass


# main code body
def main() -> None:
    # read in training prompts
    with open("prompts_reddit_train.json") as f:
        PROMPTS = json.load(f)

    DEVICE = "cuda"  # cuda/cpu/mps

    # instatiate our problem and environment
    # instatiate our problem with GPT-2 attacker, target, and baseline
    problem = HFASTProblem("gpt2", "gpt2", "gpt2", DetoxifyModerator(), DEVICE)
    env = MalibuEnv(problem, PROMPTS, gamma=0.9)

    # print out rollouts
    for i in range(1):
        rollout = env.rollout()
        print(f"ROLLOUT {i}")
        print(rollout)
        print("\n\n")

        print(f"Context: {rollout.context}")
        for t, node in enumerate(rollout.children):
            print(
                f"Turn {t}: {node.attack} | Response: {node.response} Reward: {node.reward}"
            )
        print("\n\n")


if __name__ == "__main__":
    main()
