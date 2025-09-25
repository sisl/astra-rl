from typing import Optional
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from astra_rl import ASTProblem, DetoxifyModerator, ASTEnvironment
from ast_basic import ExampleDetoxifyProblem
from astra_rl.methods.ast_problem import ASTEvaluator
from astra_rl.moderators.llamaGuard import LlamaGuardModerator


# set the attacker and target models here: we are using a pre-trained, local attacker and a "gpt2" target
# ATTACKER_MODEL = ("/home/allie11/astra-rl/examples/GPT2_v_GPT2/checkpoints/best_Detoxify_backup_GPT2")
# ATTACKER_MODEL = "gpt2"
# ATTACKER_MODEL = "/home/allie11/astra-rl/examples/GPT2_v_GPT2/checkpoints/best_Detoxify_noBackup_GPT2"
ATTACKER_MODEL = "/home/allie11/astra-rl/examples/GPT2_v_GPT2/checkpoints/best_llamaguard_noBackup_GPT2"
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

        # modify the tokenizer to account for GPT2's special fixed set up
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

        # NOTE: We inherit all the rollout/logprob methods from ExampleDetoxifyProblem.


# main code
def main() -> None:
    # set device
    DEVICE = "cuda"

    # load evaluator prompts
    with open("prompts_reddit_test.json") as f:
        PROMPTS = json.load(f)

    # instantiate your custom problem with your attacker and target models
    problem = EvaluationProblem(
        ATTACKER_MODEL, TARGET_MODEL, DEVICE, LlamaGuardModerator()
    )

    # instantiate the AST environment - no adjustments needed because already has eval_rollout
    env = ASTEnvironment(problem, PROMPTS, 1, 3)

    # instantiate the evaluator (seeds is an optional argument, must have seeds or give n_rollouts to .evaluate below)
    evaluator = ASTEvaluator(env, seeds=PROMPTS)

    # collect metrics by running n_rollouts
    metrics = evaluator.evaluate(n_rollouts=20, progress=True)

    # save metrics to json file
    evaluator.write_json(metrics, "metrics.json")


if __name__ == "__main__":
    main()
