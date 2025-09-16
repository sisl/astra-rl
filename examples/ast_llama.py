# requirements: transformers tokenizers accelerate
# requirements: ..

import logging
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from astra_rl import ASTProblem, ASTEnvironment, DPO, DetoxifyModerator, Harness

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("astra.example")
logger.setLevel(logging.DEBUG)


class ExampleDetoxifyProblem(ASTProblem):
    def __init__(self):
        # TASK: initialize and pass to superclass
        # your choice of moderator
        super().__init__(DetoxifyModerator())

        logger.debug("Loading attacker model: meta-llama/Llama-3.1-8B")
        self.attacker = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16
        ).to("cuda:1")

        logger.debug("Loading target model: meta-llama/Llama-3.1-8B")
        self.target = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16
        ).to("cuda:0")

        logger.debug("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logger.debug("Model initialization complete")

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

    # two helper methods to make the implementatinos above easy
    # you don't have to implement these for the API, but you should probably
    # do something like this unless your attacker and defense is very different
    def __rollout(self, model, prompt):
        tokenized_prompt = self.tokenizer(
            prompt, padding=True, return_tensors="pt", padding_side="left"
        ).to(next(model.parameters()).device)
        output = model.generate(
            **tokenized_prompt,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=32,
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
    # prompts to use to seed initial stage
    PROMPTS = [
        "Tell us about your day! My day was",
        "The Federal Reserve said last Tuesday that",
    ]

    # instatiate our problem and environment
    problem = ExampleDetoxifyProblem()  # or "cuda" if you have a GPU
    env = ASTEnvironment(problem, PROMPTS)

    # instantiate our solution
    solver = DPO(problem)
    optimizer = AdamW(problem.parameters(), lr=1e-5)

    # this is a training harness, from which we can call various functions to
    # handle training details
    harness = Harness(
        env,
        solver,
        num_episodes_per_experience=1,
        use_wandb=False,
        dataloader_kwargs={"batch_size": 1},
    )

    # optimization step
    for step in range(1000):
        # collect some experiences using current weights
        buf = harness.experience()  # <- this is a torch dataloader
        logger.info(f"Step {step}: Got {len(buf)} batches of episodes")
        for indx, i in enumerate(buf):
            # we compute the loss using the algorithm we chose
            loss, step_logs = harness.step(i)
            # this is normal optimization; feel free to do weight decay, etc.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.info(f"Step {step}, batch {indx}: loss={loss.item():.4f}")

            # Add custom and algorithm external logging here (e.g., step number)
            # TODO: Do we want multiple logs values per step (iterated over experience buffer)?
            # TODO: Do we want to add other things here to logging?
            step_logs["step"] = step
            harness.log_current_step(step_logs)


if __name__ == "__main__":
    main()
