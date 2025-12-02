# ASTRA-RL core components
from astra_rl import ASTSampler, LlamaGuardScorer, ASTSystem, DPO

# HuggingFace-friendly system wrapper for ASTPrompter-style red teaming
from astra_rl.training import TrainingConfiguration
from astra_rl.ext.transformers.hf_ast_system import HFASTTrainer
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# training and dev data sets - serve as initial prompts in auditor-target rollouts
from astra_rl.datasets import CONVOKIT_REDDIT_TRAIN, CONVOKIT_REDDIT_DEV

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("astra.example")
logger.setLevel(logging.WARNING)


# TODO: switch back to adv on 1 and target on 0, return gathered to 1 and switch back to .1-8B
# gpu allocation system
class MalibuSystem(ASTSystem):
    def __init__(self):
        # TASK: initialize and pass to superclass
        # your choice of scorer
        super().__init__(LlamaGuardScorer())

        logger.debug("Loading tester model: meta-llama/Llama-3.1-8B")
        self.tester = AutoModelForCausalLM.from_pretrained(
            "/home/allie11/astra-rl/examples/malibu/checkpoints/DPO_astBaseline_8B_.1b_7e-6lr/last",
            torch_dtype=torch.bfloat16,
        ).to("cuda:1")
        # "meta-llama/Llama-3.1-8B"
        logger.debug("Loading target model: meta-llama/Llama-3.1-8B")
        self.target = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16
        ).to("cuda:0")

        logger.debug("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        logger.debug("Model initialization complete")

    # TASK: you have to implement these for our API
    def get_target_logprobs(self, context, continuation):
        return self.__get_logprobs(self.target, context, continuation)

    def get_baseline_logprobs(self, context, continuation):
        # we can do this because our baseline (for KL computation)
        # and target models can be the same
        return self.get_target_logprobs(context, continuation)

    def get_tester_logprobs(self, context, continuation):
        return self.__get_logprobs(self.tester, context, continuation)

    def rollout_prompt_with_tester(self, prompt):
        return self.__rollout(self.tester, prompt)

    def rollout_prompt_with_target(self, prompt):
        return self.__rollout(self.target, prompt)

    def parameters(self):
        return self.tester.parameters()

    # two helper methods to make the implementatinos above easy
    # you don't have to implement these for the API, but you should probably
    # do something like this unless your tester and target is very different
    def __rollout(self, model, prompt):
        tokenized_prompt = self.tokenizer(
            prompt, padding=True, truncation=True, max_length=150, return_tensors="pt"
        ).to(next(model.parameters()).device)

        output = model.generate(
            **tokenized_prompt,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=24,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=1.0,
        )
        continuation = [
            i[len(j) :]
            for i, j in zip(
                self.tokenizer.batch_decode(output, skip_special_tokens=True), prompt
            )
        ]
        return continuation

    def __get_logprobs(self, model, context, continuation):
        # tokenize both context and continuation
        context = self.tokenizer(context)
        continuation = self.tokenizer(continuation)

        # create a mask such that the context is masked out
        # in order to only compute logprobs of P(continuation|context)
        combined_mask = [
            [False] * len(i) + [True] * len(j)
            for i, j in zip(context.input_ids, continuation.input_ids)
        ]

        # combine context + continuation; compute how much to pad
        combined = [i + j for i, j in zip(context.input_ids, continuation.input_ids)]
        max_length = max(len(i) for i in combined)

        # pad the combined input and context mask
        # use eos_token as padding
        combined = [
            i + [self.tokenizer.eos_token_id] * (max_length - len(i)) for i in combined
        ]
        combined_mask = [i + [False] * (max_length - len(i)) for i in combined_mask]
        attention_mask = [
            [True] * len(i) + [False] * (max_length - len(i)) for i in combined_mask
        ]

        # move things to torch and cuda
        device = next(model.parameters()).device
        combined = torch.tensor(combined).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        combined_mask = torch.tensor(combined_mask).to(device)

        # run inference
        logits = (
            model(input_ids=combined, attention_mask=attention_mask)
            .logits[:, :-1]
            .log_softmax(dim=-1)
        )

        # compute per-token likelihoods
        gathered = logits.gather(-1, combined[:, 1:].unsqueeze(-1)).squeeze(-1)
        gathered = gathered.masked_fill(~combined_mask[:, 1:], 0.0)

        # Return per-token logprobs instead of aggregating
        return gathered.to("cuda:0")
        # return gathered


def print_tree(graph, max_depth=3):
    print(f"ROOT context[:40]={graph.context[:40]!r}")
    print(f"ROOT children = {len(graph.children)}")
    for i, n0 in enumerate(graph.children):
        print(f"  [0:{i}] probe[:50]={n0.probe[:50]!r}  children={len(n0.children)}")
        if max_depth >= 2:
            for j, n1 in enumerate(n0.children):
                print(
                    f"    [1:{i}.{j}] probe[:50]={n1.probe[:50]!r}  children={len(n1.children)}"
                )
                if max_depth >= 3:
                    for k, n2 in enumerate(n1.children):
                        print(
                            f"      [2:{i}.{j}.{k}] probe[:50]={n2.probe[:50]!r}  children={len(n2.children)}"
                        )


# Help: double check training params.. match OG paper exactly
class MalibuConfiguration(TrainingConfiguration):
    def __init__(self):
        super().__init__(
            lr=0.000007,  # 1e-5
            batch_size=16,  # divide steps into batches of this size -> one optimization step per batch
            optimizer="adamw",
            gradient_accumulation_steps=16,
            training_steps=3000,  # num times collect data
            num_episodes_per_experience=1,  # num rollouts generated per experience call
        )


def main() -> None:
    # llama 3B models + llamaguard scorer -- hefty!
    # lightweight llama model: "meta-llama/Llama-3.2-1B" and DetoxifyScorer()
    # heavyweight llama model: "meta-llama/Llama-3.1-8B" and LlamaGuardScorer()
    # system = HFASTSystem(
    #     "meta-llama/Llama-3.1-8B",
    #     "meta-llama/Llama-3.1-8B",
    #     "meta-llama/Llama-3.1-8B",
    #     LlamaGuardScorer(),
    #     DEVICE,
    # )

    # newest run: set pad side during tok init and return gathered directly
    system = MalibuSystem()

    # using same reward weights as ASTPrompter paper( adv_perplexity = 0.0825, combined_weight: 2.0, defense_toxicity_weight = 0.5)
    sampler = ASTSampler(system, CONVOKIT_REDDIT_TRAIN, tree_width=2, tree_depth=3)

    # ASTPrompter used IPO
    # solver = IPO(system, beta=1.0)
    solver = DPO(system, beta=0.1)

    # set training config
    config = MalibuConfiguration()  # lr = 1e-5, batch size = 5, optimizer = "adamw", no gradient accumulation, 1000 training steps, 1 episode per experience

    # training loop is working (tested with gpt2 and saw learning)
    trainer = HFASTTrainer(
        config,
        sampler,
        solver,
        dev_prompts=CONVOKIT_REDDIT_DEV,
        eval_every=25,
        ckpt_dir="./checkpoints/DPO_astBaseline_8B_.1b_7e-6lr",
        use_wandb=True,
        show_progress=True,
    )

    # rollout = sampler.eval_rollout()
    # print(f"Full ROLLOUT")
    # print_tree(rollout, max_depth=3)
    # print("\n\n")

    # rollout2 = sampler.eval_rollout()
    # print(f"Eval ROLLOUT of size {len(rollout2)}")
    # print(rollout2)
    # print("\n\n")
    # final_reward = sampler.final_reward(rollout2)
    # print(f"Final reward: {final_reward}")
    # start training!
    trainer.train()


if __name__ == "__main__":
    main()
