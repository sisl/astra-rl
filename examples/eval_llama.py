"""
ast_basic.py
A basic example of how to use the ASTRA package.
We use GPT-2 as our attack, defense, and use the bulit-in
detoxify moderator. We will train using a manually written
corpora below of initial prompts.
"""

# requirements: transformers tokenizers
# requirements: ..

import json
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, List, Any

from astra_rl import ASTProblem, ASTEnvironment, DetoxifyModerator
from astra_rl.logging import logger


class ExampleDetoxifyProblem(ASTProblem):
    def __init__(self):
        # TASK: initialize and pass to superclass
        # your choice of moderator
        super().__init__(DetoxifyModerator())

        logger.debug("Loading attacker model: meta-llama/Llama-2-7b")
        self.attacker = AutoModelForCausalLM.from_pretrained(
            "./output/astra_llama_31_attacker/", torch_dtype=torch.bfloat16
        ).to("cuda:1")

        logger.debug("Loading target model: meta-llama/Llama-2-7b")
        self.target = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
        ).to("cuda:0")

        logger.debug("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
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


def compute_overall_metrics(rows) -> Dict[str, Any]:
    """
    Compute micro-averages across ALL turns from ALL rollouts.
    Also aggregates first-toxic-turn indices across rollouts.
    Returns a dict ready to dump as JSON.
    """

    # Flatten per-turn metrics
    all_attack_tox = [i["attack_tox"] for i in rows]
    all_target_tox = [i["target_tox"] for i in rows]
    all_attack_logprob = [i["attack_logprob"] for i in rows]

    # Means (return None if nothing to average)
    def _mean_or_none(xs: List[float]) -> Optional[float]:
        return float(sum(xs) / len(xs)) if xs else None

    metrics = {
        # Micro-averages across ALL turns from ALL rollouts
        "mean_attack_tox": _mean_or_none(all_attack_tox),
        "mean_target_tox": _mean_or_none(all_target_tox),
        "mean_attack_logprob": _mean_or_none(all_attack_logprob),
    }
    return metrics


def main():
    # read in eval prompts
    with open("/juice2/scr2/houjun/astra-rl/data/reddit/prompts_reddit_test.json") as f:
        PROMPTS = json.load(f)

    # where should we dump the logs
    out_dir = Path("./output/astra_llama_31_attacker")

    # instatiate our problem and environment
    problem = ExampleDetoxifyProblem()
    env = ASTEnvironment(problem, PROMPTS)

    rows: List[dict] = []

    for prompt in tqdm(PROMPTS):
        try:
            g = env.eval_rollout(prompt)

            rollout_rows = []
            bfs = g.children
            while len(bfs) > 0:
                node = bfs.pop(0)
                rollout_rows.append(node.to_row())
                bfs += node.children

            rows += rollout_rows
        except Exception as e:
            logger.info(f"Skipping prompt due to error: {e}")

    # Write JSON Lines (best for big datasets; one rollout per line)
    with open(out_dir / "eval_rollouts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # (Optional) also write a single JSON array for convenience
    with open(out_dir / "eval_rollouts_dump.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Global metrics across ALL rollouts
    overall = compute_overall_metrics(rows)
    print("Overall metrics:", overall)

    with open(out_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
