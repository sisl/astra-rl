"""
ast_hf.py
This is the same example as `ast.py`, but we will now use our
Huggingface extensions to enable a dramatic reduction in
boilerplate to rollouts.

We use GPT-2 as our attack, defense, and use the bulit-in
detoxify moderator. We will train using a manually written
corpora below of initial prompts.

# this example mirrors quick_start_training.py!
"""

# ASTRA-RL core components
from astra_rl import ASTEnvironment, DPO, DetoxifyModerator

# HuggingFace-friendly problem wrapper for ASTPrompter-style red teaming
from astra_rl.ext.transformers.hf_ast_problem import (
    HFASTTrainer,
    HFASTConfiguration,
    HFASTProblem,
)

# prompts to use to seed initial stage of attaker-target rollouts (training and dev)
from astra_rl.datasets import CONVOKIT_REDDIT_TRAIN, CONVOKIT_REDDIT_DEV


def main() -> None:
    DEVICE = "cuda"  # cuda/cpu/mps

    # Example problem instantiation: llama3 attacker, target, and baseline with Detoxify moderator (heavyweight setup (8B) - requires GPU)
    # problem = HFASTProblem("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", DetoxifyModerator(), DEVICE)

    # lighter weight option (1B)
    problem = HFASTProblem(
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        DetoxifyModerator(),
        DEVICE,
    )
    env = ASTEnvironment(problem, CONVOKIT_REDDIT_TRAIN)

    # instantiate our solution
    solver = DPO(problem)

    # instantiate the pre-configured HF-compatable configuration and traininer class
    config = HFASTConfiguration()  # lr = 1e-5, batch size = 4, optimizer = "adamw", no gradient accumulation, 1000 training steps, 2 episodes per experience

    # this trainer will train the attacker and evaluate it on a dev set every 100 steps, saving the best model to "checkpoints"
    trainer = HFASTTrainer(
        config,
        env,
        solver,
        dev_prompts=CONVOKIT_REDDIT_DEV,
        eval_every=100,
        ckpt_dir="checkpoints",
    )

    trainer.train()


if __name__ == "__main__":
    main()
