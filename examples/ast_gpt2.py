"""
ast_basic.py
A basic example of how to use the ASTRA package.
We use GPT-2 as our auditor, target, and use the bulit-in
detoxify scorer. We will train using a manually written
corpora below of initial prompts.
"""

# requirements: transformers tokenizers
# requirements: ..

import torch
import os
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, AutoTokenizer
from astra_rl import ASTSystem, ASTSampler, DPO, DetoxifyScorer, Harness, logger
from astra_rl.datasets import CONVOKIT_REDDIT_TRAIN, CONVOKIT_REDDIT_DEV

# MODEL_NAME = "sshleifer/tiny-gpt2" # Runs fast on cpu only
MODEL_NAME = "gpt2"


class GPT2DetoxifySystem(ASTSystem):
    def __init__(self, device="cuda"):
        # TASK: initialize and pass to superclass
        # your choice of scorer
        super().__init__(DetoxifyScorer())

        self.device = device
        self.auditor = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)
        self.target = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.auditor.config.pad_token_id = self.tokenizer.eos_token_id
        self.target.config.pad_token_id = self.tokenizer.eos_token_id

        # model’s usable max sequence length (GPT-2: 1024)
        self.max_ctx = int(
            getattr(
                self.auditor.config,
                "n_positions",
                getattr(self.auditor.config, "max_position_embeddings", 1024),
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

    def get_auditor_logprobs(self, context, continuation):
        return self.__get_logprobs(self.auditor, context, continuation)

    def rollout_prompt_with_auditor(self, prompt):
        return self.__rollout(self.auditor, prompt)

    def rollout_prompt_with_target(self, prompt):
        return self.__rollout(self.target, prompt)

    def parameters(self):
        return self.auditor.parameters()

    # two helper methods to make the implementations above easy
    # you don't have to implement these for the API, but you should probably
    # do something like this unless your auditor and target is very different
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

        # compute per-token likelihoods
        gathered = logits.gather(-1, combined[:, 1:].unsqueeze(-1)).squeeze(-1)
        gathered = gathered.masked_fill(~combined_mask[:, 1:], 0.0)

        # Return per-token logprobs instead of aggregating
        return gathered


# the following two functions will be implemented in the trainer class. This example
# does not use a trainer so we implement it here
def save(eval_sampler, step, tag="step"):
    if tag == "best":
        out = os.path.join("checkpoints", "best")  # single fixed path
    else:
        out = os.path.join("checkpoints", f"{tag}-{step}")

    # Save auditor/target in HF format
    os.makedirs(out, exist_ok=True)
    eval_sampler.system.auditor.save_pretrained(out)
    eval_sampler.system.tokenizer.save_pretrained(out)


def eval_epoch(sampler, dev_prompts, best_score, step, tag="step"):
    print(f"EVALUATING after training step {step}...")
    rewards = []

    for indx, i in enumerate(dev_prompts):
        if indx % 30 == 0:
            print(f"EVAULATED {indx}/{len(dev_prompts)} steps...")
        # perform a sigle eval rollout per dev prompt and collect a list of rewards
        # FIND WAY to extract rewards from the rollout
        rollout = sampler.eval_rollout(i)
        final_rollout_reward = sampler.final_reward(rollout)

        rewards += [final_rollout_reward]

    print(f"EVAULATED {indx}/{len(dev_prompts)} steps...")
    dev_score = sum(rewards) / len(rewards)

    if dev_score > best_score:
        logger.info(f"NEW BEST! {round(dev_score, 3)}")
        logger.info({"training/dev_score": dev_score}, step=step)
        save(sampler, step, "best")


def main() -> None:
    best_score = -float("inf")  # best score so far, used to save the best model

    DEVICE = "cuda"  # cuda/cpu/mps

    # instatiate our system and sampler
    system = GPT2DetoxifySystem(DEVICE)  # or "cuda" if you have a GPU
    sampler = ASTSampler(system, CONVOKIT_REDDIT_TRAIN)

    # instantiate our solution
    solver = DPO(system)
    optimizer = AdamW(system.parameters(), lr=1e-5)

    # this is a training harness, from which we can call various functions to
    # handle training details
    harness = Harness(
        sampler,
        solver,
        num_episodes_per_experience=2,
        use_wandb=True,
        dataloader_kwargs={"batch_size": 4},
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
            eval_epoch(sampler, CONVOKIT_REDDIT_DEV, best_score, step, "best")


if __name__ == "__main__":
    main()
