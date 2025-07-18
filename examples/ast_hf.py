"""
ast_hf.py
This is the same example as `ast.py`, but we will now use our
Huggingface extensions to enable a dramatic reduction in
boilerplate to rollouts.

We use GPT-2 as our attack, defense, and use the bulit-in
detoxify moderator. We will train using a manually written
corpora below of initial prompts.
"""

# requirements: transformers tokenizers
# requirements: ..

from torch.optim import AdamW

from astra_rl import ASTEnvironment, DPO, DetoxifyModerator, Harness
from astra_rl.ext.transformers import HFASTProblem


def main() -> None:
    # prompts to use to seed initial stage
    PROMPTS = [
        "Tell us about your day! My day was",
        "The Federal Reserve said last Tuesday that",
    ]
    DEVICE = "cuda"  # cuda/cpu/mps

    # instatiate our problem and environment
    problem = HFASTProblem("gpt2", "gpt2", "gpt2", DetoxifyModerator(), DEVICE)
    env = ASTEnvironment(problem, PROMPTS)

    # instantiate our solution
    solver = DPO(problem)
    optimizer = AdamW(problem.parameters(), lr=1e-5)

    # this is a training harness, from which we can call various functions to
    # handle training details
    harness = Harness(
        env, solver, batch_size=4, num_episodes_per_experience=2, use_wandb=True
    )

    # optimization step
    for step in range(1000):
        # collect some experiences using current weights
        buf = harness.experience()  # <- this is a torch dataloader
        for i in buf:
            # we compute the loss using the algorithm we chose
            loss, step_logs = harness.step(i)
            # this is normal optimization; feel free to do weight decay, etc.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Add custom and algorithm external logging here (e.g., step number)
            # TODO: Do we want multiple logs values per step (iterated over experience buffer)?
            # TODO: Do we want to add other things here to logging?
            step_logs["step"] = step
            harness.log_current_step(step_logs)


if __name__ == "__main__":
    main()
