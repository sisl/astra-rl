"""
ast_hf.py
This is the same example as `ast.py`, but we will now use our
Huggingface extensions to enable a dramatic reduction in
boilerplate to rollouts.

We use GPT-2 as our tester, target, and use the bulit-in
detoxify scorer. We will train using a manually written
corpora below of initial prompts.

# this example mirrors quick_start_training.py!
"""

# ASTRA-RL core components
from astra_rl import ASTSampler, DPO, DetoxifyScorer

# HuggingFace-friendly system wrapper for AST-style red teaming
from astra_rl.ext.transformers.hf_ast_system import (
    HFASTTrainer,
    HFASTConfiguration,
    HFASTSystem,
)

# prompts to use to seed initial stage of tester-target rollouts (training and dev)
from astra_rl.datasets import CONVOKIT_REDDIT_TRAIN, CONVOKIT_REDDIT_DEV


def main() -> None:
    DEVICE = "cuda"  # cuda/cpu/mps

    # Example system instantiation: llama3 tester, target, and baseline with Detoxify scorer (heavyweight setup (8B) - requires GPU)
    # system = HFASTSystem("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", DetoxifyScorer(), DEVICE)

    # lighter weight option (1B)
    system = HFASTSystem(
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        DetoxifyScorer(),
        DEVICE,
    )

    # make sure all prompts are at most 500 characters (we don't want huge paragraphs, will oom)
    PROMPTS = [p[:500] for p in CONVOKIT_REDDIT_TRAIN]
    sampler = ASTSampler(system, PROMPTS)

    # instantiate our solution
    solver = DPO(system)

    # instantiate the pre-configured HF-compatable configuration and traininer class
    config = HFASTConfiguration()  # lr = 7e-6, batch size = 16, optimizer = "adamw", 16 gradient accumulation steps, 3000 training steps, 1 episode per experience

    # this trainer will train the tester and evaluate it on a dev set every 100 steps, saving the best model to "./checkpoints/huggingface"
    trainer = HFASTTrainer(
        config,
        sampler,
        solver,
        dev_prompts=CONVOKIT_REDDIT_DEV,
        eval_every=25,
        ckpt_dir="./checkpoints/custom_name",
        use_wandb=True,
        show_progress=True,
    )

    trainer.train()


if __name__ == "__main__":
    main()
