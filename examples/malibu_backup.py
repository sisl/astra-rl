"""
malibu.py
This uses the ast environment and Huggingface extensions. However, with modifications to the environment,
we can enable the backup of discounted future rewards during multi-turn conversation rollouts used in DPO.

This example shows how to use the Astra RL library to train a GPT-2 model
using the DPO algorithm with a Detoxify moderator for content moderation.

We will train using the ConvoKit Reddit Small Corpa for initial prompts, filtered by the moderator to be non-toxic.
"""

# requirements: transformers tokenizers
# requirements: ..

from torch.optim import AdamW
from astra_rl import ASTEnvironment, DPO, DetoxifyModerator, Harness
from astra_rl.ext.transformers import HFASTProblem

# prompts to use to seed initial stage
PROMPTS = [
    "Tell us about your day! My day was",
    "The Federal Reserve said last Tuesday that",
]

# instatiate our problem and environment
problem = HFASTProblem("gpt2", "gpt2", "gpt2", DetoxifyModerator(), "cuda")

# malibu magic time! Create a custom environment
env = ASTEnvironment(problem, PROMPTS)

# instantiate our solution
solver = DPO(problem)
optimizer = AdamW(problem.parameters(), lr=1e-5)

# this is a training harness, from which we can call various functions to
# handle training details
harness = Harness(env, solver, batch_size=4, num_episodes_per_experience=2)

# optimization step
for _ in range(1000):
    # collect some experiences using current weights
    buf = harness.experience()  # <- this is a torch dataloader
    for i in buf:
        # we compute the loss using the algorithm we chose
        loss = harness.step(i)
        # this is normal optimization; feel free to do weight decay, etc.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
