from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from astra_rl import ASTProblem, DetoxifyModerator, ASTEnvironment
from astra_rl.datasets import CONVOKIT_REDDIT_TEST
from ast_gpt2 import GPT2DetoxifyProblem
from astra_rl.methods.ast_problem import ASTEvaluator
from astra_rl.moderators.llamaGuard import LlamaGuardModerator

# change name to length-limited eval

# set the attacker model here: we are using a pre-trained, local attacker and a "gpt2" target
ATTACKER_MODEL = "/home/allie11/astra-rl/examples/GPT2_v_GPT2/checkpoints/best_llamaguard_noBackup_GPT2"


# since we are using a custom attacker that is not supported by HFASTProblem (GPT2 is not supported)
# we must create a custom subclass of GPT2DetoxifyProblem
class GPT2EvaluationProblem(GPT2DetoxifyProblem):
    """
    Same API/behavior as ExampleDetoxifyProblem, but lets you plug in a custom attacker
    (local HF directory or hub id) and provide a custom moderator.

    Args:
        attacker_model (str): The model ID or path for the attacker model, must be possible
            for AutoModelForCausalLM.
        device (str): The device to run the models on, e.g. "cpu"
        moderator (Optional[DetoxifyModerator]): The moderator to use, defaults to Detoxify.

    Note:
        This class inherits all methods from ExampleDetoxifyProblem except for __init__,
        which is modified to allow for a custom attacker model and moderator.

        ** Assumes target is "gpt2" and that the attacker is a an adversarial model based on gpt2.
    """

    def __init__(
        self,
        attacker_model: str = ATTACKER_MODEL,
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
        self.target = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

        # Tokenizer is shared between attacker and target (both gpt2-based)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

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

    # instantiate your custom problem with your attacker and target models
    problem = GPT2EvaluationProblem(ATTACKER_MODEL, DEVICE, LlamaGuardModerator())

    # instantiate the AST environment - no adjustments needed because already has eval_rollout
    env = ASTEnvironment(problem, CONVOKIT_REDDIT_TEST, 1, 3)

    # instantiate the evaluator (seeds is an optional argument, must have seeds or give n_rollouts to .evaluate below)
    evaluator = ASTEvaluator(env, seeds=CONVOKIT_REDDIT_TEST)

    # collect metrics by running n_rollouts
    metrics = evaluator.evaluate(n_rollouts=20, progress=True)

    # save metrics to json file
    evaluator.write_json(metrics, "metrics.json")


if __name__ == "__main__":
    main()
