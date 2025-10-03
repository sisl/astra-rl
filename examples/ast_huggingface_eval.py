# import dependencies
import torch
from astra_rl import DetoxifyScorer, ASTSampler
from astra_rl.datasets import CONVOKIT_REDDIT_TEST
from astra_rl.methods.ast_system import ASTEvaluator
from astra_rl.ext.transformers.hf_ast_system import HFEvaluationSystem


def main() -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Path to your pre-trained llama3 tester model
    TESTER_MODEL = "./checkpoints/huggingface/best"  # assuming tokenizer is in checkpoint (default save in training)
    TARGET_MODEL = "meta-llama/Llama-3.1-8B"  # can be any HF model

    # instantiate the HF evaluation system with your tester and target models
    # here we do not give the optional TESTER_BASE_MODEL arg becuase the tester checkpoint has tokenizer info in it
    # if your checkpoint does not have tokenizer info, you must provide the base model id (e.g. "meta-llama/Llama-3.1-8B")
    # system = HFEvaluationSystem(tester_checkpoint=TESTER_MODEL, tester_base_model_id=None, target_model_id=TARGET_MODEL,device=DEVICE, scorer=DetoxifyScorer())
    system = HFEvaluationSystem(
        tester_checkpoint=TESTER_MODEL,
        tester_base_model_id=None,
        target_model_id=TARGET_MODEL,
        device=DEVICE,
        scorer=DetoxifyScorer(),
    )
    # instantiate the AST sampler - no adjustments needed because already has eval_rollout
    sampler = ASTSampler(system, CONVOKIT_REDDIT_TEST, 1, 3)

    # instantiate the evaluator
    # note: seeds is an optional argument, must have seeds here or give n_rollouts to .evaluate below, we do the latter here
    evaluator = ASTEvaluator(sampler, seeds=CONVOKIT_REDDIT_TEST)

    # collect metrics by running n_rollouts
    metrics = evaluator.evaluate(n_rollouts=2, progress=True)

    # save metrics to json file
    evaluator.write_json(metrics, "example_hf_eval_metrics.json")


if __name__ == "__main__":
    main()
