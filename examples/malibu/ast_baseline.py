# ASTRA-RL core components
from astra_rl import ASTSampler, IPO, DetoxifyScorer

# HuggingFace-friendly system wrapper for ASTPrompter-style red teaming
from astra_rl.ext.transformers import HFASTSystem
from astra_rl.training import TrainingConfiguration
from astra_rl.ext.transformers.hf_ast_system import HFASTTrainer

# training and dev data sets - serve as initial prompts in auditor-target rollouts
from astra_rl.datasets import CONVOKIT_REDDIT_TRAIN, CONVOKIT_REDDIT_DEV


# Help: double check training params.. slight differences from OG paper
class MalibuConfiguration(TrainingConfiguration):
    def __init__(self):
        super().__init__(
            lr=1e-5,
            batch_size=5,  # divide steps into batches of this size -> one optimization step per batch
            optimizer="adamw",
            gradient_accumulation_steps=1,
            training_steps=1000,  # num times collect data
            num_episodes_per_experience=1,  # num rollouts generated per experience call
        )


def main() -> None:
    DEVICE = "cuda"  # or "cpu" if GPU is not available

    # llama 3B models + llamaguard scorer -- hefty!
    # lightweight llama model: "meta-llama/Llama-3.2-1B" and DetoxifyScorer()
    # heavyweight llama model: "meta-llama/Llama-3.1-8B" and LlamaGuardScorer()
    system = HFASTSystem(
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        DetoxifyScorer(),
        DEVICE,
    )

    # We are rollout out the convo one step longer since we are focusing on long-range attack strategies
    sampler = ASTSampler(system, CONVOKIT_REDDIT_TRAIN, tree_width=2, tree_depth=4)

    # ASTPrompter used IPO
    solver = IPO(system)

    # set training config
    config = MalibuConfiguration()  # lr = 1e-5, batch size = 5, optimizer = "adamw", no gradient accumulation, 1000 training steps, 1 episode per experience

    trainer = HFASTTrainer(
        config,
        sampler,
        solver,
        dev_prompts=CONVOKIT_REDDIT_DEV,
        eval_every=100,
        ckpt_dir="./checkpoints/huggingface",
        use_wandb=True,
        show_progress=True,
    )

    # start training!
    trainer.train()


if __name__ == "__main__":
    main()
