from typing import Optional
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from astra_rl import ASTProblem, DetoxifyModerator, ASTEnvironment

# How to import ExampleDetoxifyProblem
from ast_basic import ExampleDetoxifyProblem
from astra_rl.core.evaluator import Evaluator

# set the attacker and target models here: we are using a pre-trained, local attacker and a "gpt2" target
ATTACKER_MODEL = (
    "/home/allie11/astra-rl/examples/GPT2_v_GPT2/checkpoints/best_Detoxify_backup_GPT2"
)
TARGET_MODEL = "gpt2"


# since we are using a custom attacker that is not supported by HFASTProblem (GPT2 is not supported)
# we must quickly create a custom subclass of ExampleDetoxifyProblem
class EvaluationProblem(ExampleDetoxifyProblem):
    """
    Same API/behavior as ExampleDetoxifyProblem, but lets you plug in a custom attacker
    (local HF directory or hub id) and optional target + moderator.
    """

    def __init__(
        self,
        attacker_model: str = ATTACKER_MODEL,
        target_model: str = TARGET_MODEL,
        device: str = "cpu",
        moderator: Optional[DetoxifyModerator] = None,
    ):
        # Bypass ExampleDetoxifyProblem.__init__ (it hardcodes models/moderator).
        # Initialize the ASTProblem base with given moderator (optional, defaults to Detoxify).
        ASTProblem.__init__(self, moderator or DetoxifyModerator())

        self.device = device

        # Load attacker
        self.attacker = AutoModelForCausalLM.from_pretrained(attacker_model).to(
            self.device
        )

        # Load target (gpt2 by default to match ExampleDetoxifyProblem)
        self.target = GPT2LMHeadModel.from_pretrained(target_model).to(self.device)

        # Tokenizer is shared between attacker and target (both gpt2-based)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # NOTE: We inherit all the rollout/logprob methods from ExampleDetoxifyProblem.


# main code
def main() -> None:
    # set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # load evaluator prompts
    with open("prompts_reddit_test.json") as f:
        PROMPTS = json.load(f)

    # instantiate your custom problem with your attacker and target models
    problem = EvaluationProblem(
        attacker_model_id=ATTACKER_MODEL,
        target_model_id=TARGET_MODEL,
        moderator=DetoxifyModerator(),
        device=DEVICE,
    )

    # instantiate the AST environment - no adjustments needed because already has eval_rollout
    env = ASTEnvironment(
        problem=problem,
        prompts=PROMPTS,
        tree_width=1,
        tree_depth=3,
    )

    # instantiate the evaluator
    evaluator = Evaluator(env, seeds=PROMPTS)

    # collect metrics by running n_rollouts
    metrics = evaluator.evaluate(n_rollouts=10)

    # save metrics to json file
    Evaluator.write_json(metrics, "metrics.json")


if __name__ == "__main__":
    main()
