from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from astra_rl import ASTSystem, DetoxifyScorer, ASTSampler
from astra_rl.datasets import CONVOKIT_REDDIT_TEST
from ast_gpt2 import GPT2DetoxifySystem
from astra_rl.methods.ast_system import ASTEvaluator
from astra_rl.scorers.llamaguard import LlamaGuardScorer

# change name to length-limited eval

# set the auditor model here: we are using a pre-trained, local auditor and a "gpt2" target
AUDITOR_MODEL = "/home/allie11/astra-rl/examples/GPT2_v_GPT2/checkpoints/best_llamaguard_noBackup_GPT2"


# since we are using a custom auditor that is not supported by HFASTSystem (GPT2 is not supported)
# we must create a custom subclass of GPT2DetoxifySystem
class GPT2EvaluationSystem(GPT2DetoxifySystem):
    """
    Same API/behavior as ExampleDetoxifySystem, but lets you plug in a custom auditor
    (local HF directory or hub id) and provide a custom scorer.

    Args:
        auditor_model (str): The model ID or path for the auditor model, must be possible
            for AutoModelForCausalLM.
        device (str): The device to run the models on, e.g. "cpu"
        scorer (Optional[DetoxifyScorer]): The scorer to use, defaults to Detoxify.

    Note:
        This class inherits all methods from ExampleDetoxifySystem except for __init__,
        which is modified to allow for a custom auditor model and scorer.

        ** Assumes target is "gpt2" and that the auditor is an adversarial model based on gpt2.
    """

    def __init__(
        self,
        auditor_model: str = AUDITOR_MODEL,
        device: str = "cpu",
        scorer: Optional[DetoxifyScorer] = None,
    ):
        # Bypass ExampleDetoxifySystem.__init__ (it hardcodes models/scorer).
        # Initialize the ASTSystem base with given scorer (optional, defaults to Detoxify).
        ASTSystem.__init__(self, scorer or DetoxifyScorer())

        self.device = device

        # Load auditor
        self.auditor = AutoModelForCausalLM.from_pretrained(auditor_model).to(
            self.device
        )

        # Load target (gpt2 by default to match ExampleDetoxifySystem)
        self.target = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

        # Tokenizer is shared between auditor and target (both gpt2-based)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # modify the tokenizer to account for GPT2's special fixed set up
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

        # NOTE: We inherit all the rollout/logprob methods from ExampleDetoxifySystem.


# main code
def main() -> None:
    # set device
    DEVICE = "cuda"

    # instantiate your custom system with your auditor and target models
    system = GPT2EvaluationSystem(AUDITOR_MODEL, DEVICE, LlamaGuardScorer())

    # instantiate the AST sampler - no adjustments needed because already has eval_rollout
    sampler = ASTSampler(system, CONVOKIT_REDDIT_TEST, 1, 3)

    # instantiate the evaluator (seeds is an optional argument, must have seeds or give n_rollouts to .evaluate below)
    evaluator = ASTEvaluator(sampler, seeds=CONVOKIT_REDDIT_TEST)

    # collect metrics by running n_rollouts
    metrics = evaluator.evaluate(n_rollouts=20, progress=True)

    # save metrics to json file
    evaluator.write_json(metrics, "metrics.json")


if __name__ == "__main__":
    main()
