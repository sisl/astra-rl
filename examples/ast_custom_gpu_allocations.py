"""
ast_llama.py
An example of using AST with LLaMA models as tester and target where
the GPU allocations are explicitly specified.
"""

# requirements: transformers tokenizers accelerate
# requirements: ..

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from astra_rl.ext.transformers.hf_ast_system import HFASTTrainer, HFASTConfiguration
from astra_rl import ASTSystem, ASTSampler, DPO, LlamaGuardScorer

# training and dev data sets - serve as initial prompts in tester-target rollouts
from astra_rl.datasets import CONVOKIT_REDDIT_TRAIN, CONVOKIT_REDDIT_DEV


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("astra.example")
logger.setLevel(logging.DEBUG)


# gpu allocation system
class GPUAllocationSystem(ASTSystem):
    def __init__(self):
        # TASK: initialize and pass to superclass
        # your choice of scorer
        super().__init__(LlamaGuardScorer())

        logger.debug("Loading tester model: meta-llama/Llama-3.1-8B")
        self.tester = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
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
        tokenized_prompt = self.tokenizer(prompt, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

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


def main() -> None:
    # prompts to use to seed initial stage - this is a tiny example of two prompts
    # PROMPTS = [
    #     "Tell us about your day! My day was",
    #     "The Federal Reserve said last Tuesday that",
    # ]
    # full training dataset - convokit reddit small
    PROMPTS = CONVOKIT_REDDIT_TRAIN
    # make sure all prompts are at most 500 characters (we don't want huge paragraphs, will oom)
    PROMPTS = [p[:500] for p in PROMPTS]

    # instatiate our system and sampler
    system = GPUAllocationSystem()
    sampler = ASTSampler(system, PROMPTS)

    # instantiate our solution
    solver = DPO(system)

    config = HFASTConfiguration()

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
