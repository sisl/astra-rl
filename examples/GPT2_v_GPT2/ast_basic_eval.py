from transformers import AutoTokenizer
from typing import Optional, List
import torch
import json
from transformers import GPT2LMHeadModel
from astra_rl import ASTProblem, ASTEnvironment, DetoxifyModerator
from ast_basic_1 import ExampleDetoxifyProblem
from statistics import mean
from typing import Dict, Any


ATTACKER_MODEL_PATH = "/home/allie11/astra-rl/examples/GPT2_v_GPT2/checkpoints/best"
TARGET_MODEL = "gpt2"


# you will need to create a new subclass of ASTProblem that uses the trained attacker model
# Since we are evaluating the model we trained in ast_basic , we will subclass from ExampleDetoxifyProblem
# and override the init method to load the trained attacker model. Everything else will remain the same.
# Note: make sure the target model you want to evaluate with is the same as the one used
# in the problem ldefinition for get log probs!
class ExampleEvalProblem(ExampleDetoxifyProblem):
    def __init__(self, device="cuda"):
        ASTProblem.__init__(self, DetoxifyModerator())

        self.device = device
        self.attacker = GPT2LMHeadModel.from_pretrained(ATTACKER_MODEL_PATH).to(
            self.device
        )
        self.target = GPT2LMHeadModel.from_pretrained(TARGET_MODEL).to(self.device)
        self.attacker.eval()
        self.target.eval()

        # since we use the same tokenizer for both attacker and target, we can just load it once
        self.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
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
        print(f"Using model {TARGET_MODEL} with max context length {self.max_ctx}")


# helper function to turn an eval rollout with metrics into a dictionary
def rollout_to_row(prompt: str, summary, tokenizer) -> dict:
    """
    Turn an EvalSummary into a single flat dict row.
    Includes attack_i/response_i and rollout-level stats.
    """
    turns = sorted(summary.turns, key=lambda t: t.turn_index)  # ensure attack_1..N
    row = {
        "prompt": prompt,
        "first_tox_attack": summary.first_attack_tox_turn,
        "first_tox_response": summary.first_target_tox_turn,
    }

    # Per-turn fields (attack_i, response_i)
    for i, t in enumerate(turns, start=1):
        row[f"attack_{i}"] = t.attack
        row[f"response_{i}"] = t.response

    # Aggregates
    if turns:
        mean_attack_tox = sum(t.attack_tox for t in turns) / len(turns)
        mean_response_tox = sum(t.target_tox for t in turns) / len(turns)
        mean_attack_logprob = sum(t.attack_logprob for t in turns) / len(turns)

    else:
        mean_attack_tox = 0.0
        mean_response_tox = 0.0
        mean_attack_logprob = 0.0

    row["mean_attack_tox"] = mean_attack_tox
    row["mean_response_tox"] = mean_response_tox
    row["mean_attack_logprob"] = mean_attack_logprob

    return row


def summaries_to_overall_metrics(summaries) -> Dict[str, Any]:
    """
    Compute micro-averages across ALL turns from ALL rollouts.
    Also aggregates first-toxic-turn indices across rollouts.
    Returns a dict ready to dump as JSON.
    """
    # Flatten per-turn metrics
    all_attack_tox = [t.attack_tox for s in summaries for t in s.turns]
    all_target_tox = [t.target_tox for s in summaries for t in s.turns]
    all_attack_logprob = [t.attack_logprob for s in summaries for t in s.turns]

    # First toxic turn indices (None means "never")
    first_attack_turns_present = [
        s.first_attack_tox_turn
        for s in summaries
        if s.first_attack_tox_turn is not None
    ]
    first_target_turns_present = [
        s.first_target_tox_turn
        for s in summaries
        if s.first_target_tox_turn is not None
    ]

    num_rollouts = len(summaries)
    attack_cov = (
        (len(first_attack_turns_present) / num_rollouts) if num_rollouts else 0.0
    )
    target_cov = (
        (len(first_target_turns_present) / num_rollouts) if num_rollouts else 0.0
    )

    # Means (return None if nothing to average)
    def _mean_or_none(xs: List[float]) -> Optional[float]:
        return float(mean(xs)) if xs else None

    metrics = {
        # Micro-averages across ALL turns from ALL rollouts
        "mean_attack_tox": _mean_or_none(all_attack_tox),
        "mean_target_tox": _mean_or_none(all_target_tox),
        "mean_attack_logprob": _mean_or_none(all_attack_logprob),
        # First-toxic-turn stats across rollouts
        # (mean index among rollouts where toxicity ever occurred)
        "mean_first_tox_attack_turn": _mean_or_none(first_attack_turns_present),
        "mean_first_tox_response_turn": _mean_or_none(first_target_turns_present),
        # Coverage: fraction of rollouts that ever became toxic (attack/response)
        "attack_tox_coverage(assuming toxic)": attack_cov,
        "response_tox_coverage(assuming toxic)": target_cov,
        # (Optional) means including missing as 0 if you prefer that interpretation
        "mean_first_tox_attack_turn_including_missing_as_0": _mean_or_none(
            [(s.first_attack_tox_turn or 0) for s in summaries]
        ),
        "mean_first_tox_response_turn_including_missing_as_0": _mean_or_none(
            [(s.first_target_tox_turn or 0) for s in summaries]
        ),
    }
    return metrics


def main():
    # read in eval prompts
    with open("prompts_reddit_test.json") as f:
        PROMPTS = json.load(f)

    # set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # instatiate our problem and environment
    problem = ExampleEvalProblem(DEVICE)
    env = ASTEnvironment(problem, PROMPTS)

    rows: List[dict] = []
    summaries = []

    for prompt in PROMPTS:
        g = env.eval_rollout(prompt)
        summary = env.extract_metrics(g)  # EvalSummary(turns=[...], first_*_turn=...)
        summaries.append(summary)
        row = rollout_to_row(prompt, summary, env.problem.tokenizer)
        rows.append(row)
        # print(f"Evaluated prompt: {prompt}, Summary: {summary}")

    # Write JSON Lines (best for big datasets; one rollout per line)
    with open("eval_rollouts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # (Optional) also write a single JSON array for convenience
    with open("eval_rollouts_dump.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Global metrics across ALL rollouts
    overall = summaries_to_overall_metrics(summaries)
    print("Overall metrics:", overall)

    with open("overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
