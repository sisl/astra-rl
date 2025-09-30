# import dependencies
import torch
import json
from astra_rl import DetoxifyModerator, ASTEnvironment
from astra_rl.methods.ast_problem import ASTEvaluator
from astra_rl.ext.transformers.hf_ast_problem import HFEvaluationProblem


def main() -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to your pre-trained llama3 attacker model
    ATTACKER_MODEL = (
        "meta-llama/Llama-3.1-8B"  # e.g., "/home/checkpoints/best" or "gpt2"
    )
    ATTACKER_BASE_MODEL = "meta-llama/Llama-3.1-8B"  # assuming you used a llama3 model as the attacker during training
    TARGET_MODEL = "meta-llama/Llama-3.1-8B"  # can be any HF model

    with open("prompts_reddit_test.json") as f:
        PROMPTS = json.load(f)  # e.g., ["prompt 1", "prompt 2", ...]

    # instantiate the HF evaluation problem with your attacker and target models
    problem = HFEvaluationProblem(
        ATTACKER_MODEL, ATTACKER_BASE_MODEL, TARGET_MODEL, DEVICE, DetoxifyModerator()
    )

    # instantiate the AST environment - no adjustments needed because already has eval_rollout
    env = ASTEnvironment(problem, PROMPTS, 1, 3)

    # instantiate the evaluator
    # note: seeds is an optional argument, must have seeds here or give n_rollouts to .evaluate below, we do the latter here
    evaluator = ASTEvaluator(env, seeds=PROMPTS)

    # collect metrics by running n_rollouts
    metrics = evaluator.evaluate(n_rollouts=2, progress=True)

    # save metrics to json file
    evaluator.write_json(metrics, "naive_llama_metrics.json")


if __name__ == "__main__":
    main()
