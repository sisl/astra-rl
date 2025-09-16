"""
ast_basic.py
A basic example of how to use the ASTRA package.
We use GPT-2 as our attack, defense, and use the bulit-in
detoxify moderator. We will train using a manually written
corpora below of initial prompts.
"""

# requirements: transformers tokenizers
# requirements: ..

import torch
import json
from transformers import GPT2LMHeadModel, AutoTokenizer
from astra_rl.ext.transformers.hf_ast_problem import (
    HFASTTrainer,
    HFASTConfiguration,
)

from astra_rl import ASTProblem, ASTEnvironment, DPO
from astra_rl.moderators.llamaGuard import LlamaGuardModerator

# MODEL_NAME = "sshleifer/tiny-gpt2" # Runs fast on cpu only
MODEL_NAME = "gpt2"


class ExampleLlamaGuardProblem(ASTProblem):
    def __init__(self, device="cuda"):
        # TASK: initialize and pass to superclass
        # one line of code difference to call a different moderator!!
        super().__init__(LlamaGuardModerator(device))

        self.device = device
        self.attacker = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)
        self.target = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.attacker.config.pad_token_id = self.tokenizer.eos_token_id
        self.target.config.pad_token_id = self.tokenizer.eos_token_id

        # model’s usable max sequence length (GPT-2: 1024)
        self.max_ctx = int(
            getattr(
                self.attacker.config,
                "n_positions",
                getattr(self.attacker.config, "max_position_embeddings", 1024),
            )
        )
        print(f"Using model {MODEL_NAME} with max context length {self.max_ctx}")

    # TASK: you have to implement these for our API
    def get_target_logprobs(self, context, continuation):
        return self.__get_logprobs(self.target, context, continuation)

    def get_baseline_logprobs(self, context, continuation):
        # we can do this because our baseline (for KL computation)
        # and target models can be the same
        return self.get_target_logprobs(context, continuation)

    def get_attacker_logprobs(self, context, continuation):
        return self.__get_logprobs(self.attacker, context, continuation)

    def rollout_prompt_with_attacker(self, prompt):
        return self.__rollout(self.attacker, prompt)

    def rollout_prompt_with_target(self, prompt):
        return self.__rollout(self.target, prompt)

    def parameters(self):
        return self.attacker.parameters()

    # two helper methods to make the implementations above easy
    # you don't have to implement these for the API, but you should probably
    # do something like this unless your attacker and defense is very different
    def __rollout(self, model, prompt):
        gen_length = 32
        max_context_len = self.max_ctx - gen_length
        # we truncate the prompt to 1024 - 32 tokens to avoid a PyTorch CUDA device-side indexing error (conversation contexts can get too long in the multiturn setting)
        tokenized_prompt = self.tokenizer(
            prompt,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_context_len,
            add_special_tokens=False,  # I added this, is it okay?
        ).to(self.device)

        # print statments to find bug
        ids = tokenized_prompt["input_ids"]
        seq_len = ids.shape[1]
        # print("ROLL seq_len:", seq_len, "max_new:", 32, "total_if_generated:", seq_len + 32)
        assert seq_len + 32 <= getattr(model.config, "n_positions", 1024)

        output = model.generate(
            **tokenized_prompt,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=gen_length,
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
        # make sure context is not too long (context + continuation should be <= 1024 / max seq len for GPT2)
        context = self.tokenizer(context)
        # continuation should be only 32 tokens long
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

        # move things to torch and cuda (make sure indicies <= 1024 for GPT2... this is model specific!)
        # TODO: show how to make this capping flexible to the model to help future users
        combined = torch.tensor(combined).to(self.device)[
            :, -self.max_ctx :
        ]  # cap length to 1024
        attention_mask = torch.tensor(attention_mask).to(self.device)[
            :, -self.max_ctx :
        ]
        combined_mask = torch.tensor(combined_mask).to(self.device)[:, -self.max_ctx :]

        # print statements to find bug
        # print("LOGPROBS seq_len:", combined.shape[1])
        assert combined.shape[1] <= getattr(model.config, "n_positions", 1024)

        # run inference
        logits = (
            model(input_ids=combined, attention_mask=attention_mask)
            .logits[:, :-1]
            .log_softmax(dim=-1)
        )

        # compute likelihoods
        gathered = logits.gather(-1, combined[:, 1:].unsqueeze(-1)).squeeze(-1)
        gathered = gathered.masked_fill(~combined_mask[:, 1:], 0.0)
        logprobs = gathered.sum(dim=-1)

        return logprobs


def main() -> None:
    # prompts to use to seed initial stage
    # read in training prompts
    with open("prompts_reddit_train.json") as f:
        PROMPTS = json.load(f)

    # read in dev set of prompts
    with open("prompts_reddit_dev.json") as f:
        DEV_PROMPTS = json.load(f)

    DEVICE = "cuda"  # cuda/cpu/mps

    # instatiate our problem and environment
    problem = ExampleLlamaGuardProblem(DEVICE)  # or "cuda" if you have a GPU
    env = ASTEnvironment(problem, PROMPTS)

    # instantiate our solution
    solver = DPO(problem)

    # instantiate the pre-configured HF-compatable traininer class
    # Source code is in src/astra_rl/ext/transformers/hf_ast_problem.py (shows training hyperparams and how checkpointing/saving models)
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
