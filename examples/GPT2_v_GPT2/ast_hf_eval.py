from typing import List
import torch
import json
from astra_rl.ext.transformers import HFASTProblem
from astra_rl import ASTEnvironment, DetoxifyModerator, CustomNode
from ast_basic_eval import summaries_to_overall_metrics
from statistics import mean


ATTACKER_MODEL_PATH = "/home/allie11/astra-rl/examples/GPT2_v_GPT2/checkpoints/best"
TARGET_MODEL = "gpt2"


def main():
    # read in eval prompts (should be prompts the attacker has never seen (not training or dev prompts))
    with open("prompts_reddit_test.json") as f:
        PROMPTS = json.load(f)

    # set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # instatiate our problem and environment
    problem = HFASTProblem(
        ATTACKER_MODEL_PATH, TARGET_MODEL, TARGET_MODEL, DetoxifyModerator(), DEVICE
    )
    env = ASTEnvironment(problem, PROMPTS)

    rows: List[dict] = []
    summaries = []

    for prompt in PROMPTS:
        g, first_attack_tox, first_target_tox = env.eval_rollout(prompt)
        summaries.append(
            CustomNode(
                context=prompt,
                attack="",
                response="",
                reward=problem.final_reward(g) or 0.0,
                children=g.children[0].children,
                attack_tox=mean([n.attack_tox for n in g.children[0].children])
                if g.children[0].children
                else 0.0,
                target_tox=mean([n.target_tox for n in g.children[0].children])
                if g.children[0].children
                else 0.0,
                attack_logprob=mean([n.attack_logprob for n in g.children[0].children])
                if g.children[0].children
                else 0.0,
                first_attack_tox_turn=first_attack_tox
                if first_attack_tox != -1
                else None,
                first_target_tox_turn=first_target_tox
                if first_target_tox != -1
                else None,
            )
        )
        row = [n.to_row() for n in g.children[0].children]
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
