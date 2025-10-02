# Evaluation Problems

When performing an **evaluation**, you need to create a *problem class* that correctly loads your **trained attacker model** and its tokenizer.

Most of the time, the only change required is **how the attacker model and tokenizer are instantiated**. All rollout logic and evaluation APIs remain the same.

---

## 1. Using Hugging Face Models (non-GPT2)

If your trained attacker is a Hugging Face model that does **not** have a fixed maximum context length (e.g. LLaMA-3), you can simply use `HFEvaluationProblem`.

```python
from astra_rl.ext.transformers import HFEvaluationProblem

problem = HFEvaluationProblem(
    attacker_model="/path/to/your/attacker/checkpoint",
    attacker_base_model_id="meta-llama/Meta-Llama-3-8B",  # base tokenizer
    target_model_id="meta-llama/Meta-Llama-3-8B",
    device="cuda"
)
```

!!! tip
    See the [Quick Start: Evaluation](../quick_start_evaluation.md) tutorial or the [full HF evaluation example](https://github.com/sisl/astra-rl/tree/main/examples/hf_eval.py).

---

## 2. Using GPT-2–based Attackers

GPT-2 has some quirks (fixed max length of 1024, special padding setup).
We provide [`GPT2EvaluationProblem`](ttps://github.com/sisl/astra-rl/tree/main/examples/gpt2_eval.py), which handles this automatically:

```python
from gpt2_eval import GPT2EvaluationProblem

problem = GPT2EvaluationProblem(
    attacker_model="/path/to/attacker/checkpoint",
    device="cuda"
)
```

**Key details:**

* Inherits from `GPT2DetoxifyProblem`.
* Only overrides `__init__` to let you pass in a custom attacker and moderator.
* Assumes:

  * Target = `"gpt2"`
  * Attacker = GPT-2–based adversarial model.

!!! tip
    See the [full GPT2 evaluation example](https://github.com/sisl/astra-rl/blob/main/examples/gpt2_eval.py).
---

## 3. Fully Custom Attackers or Targets

If you are using a **completely custom pre-trained attacker or target**, you will need to define your own subclass of `ASTProblem`.
This subclass must:

1. Instantiate attacker, target, and tokenizers.
2. Implement rollout logic (text generation given context).

See the [Problem Customization guide](../customizing_training/problems.md) for details.

!!! note
    If you already created a custom problem class for **training**, it is often easiest to **subclass it for evaluation** and just modify the attacker instantiation.

    For example, the [`GPT2EvaluationProblem`](https://github.com/sisl/astra-rl/blob/main/examples/gpt2_eval.py) is a thin subclass that changes only the constructor.

---

## 4. Example: GPT-2 Custom Evaluation Problem

Here’s a concrete example showing how to create a custom problem that loads a trained GPT-2 attacker and a standard GPT-2 target:

```python

ATTACKER_MODEL = "path/to/your/attacker/checkpoint"

class GPT2EvaluationProblem(GPT2DetoxifyProblem):
    """
    Same API/behavior as GPT2DetoxifyProblem, but with a custom attacker and moderator.
    Assumes target is GPT-2.
    """

    def __init__(self, attacker_model: str = ATTACKER_MODEL,
                 device: str = "cpu",
                 moderator: Optional[DetoxifyModerator] = None):
        ASTProblem.__init__(self, moderator or DetoxifyModerator())
        self.device = device

        # Attacker (trained GPT-2 adversary)
        self.attacker = AutoModelForCausalLM.from_pretrained(attacker_model).to(device)

        # Target (plain GPT-2)
        self.target = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        # Shared tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.attacker.config.pad_token_id = self.tokenizer.eos_token_id
        self.target.config.pad_token_id = self.tokenizer.eos_token_id

        # Max context length
        self.max_ctx = int(getattr(self.attacker.config, "n_positions",
                                   getattr(self.attacker.config, "max_position_embeddings", 1024)))
```

---

## 5. Putting It All Together

After creating your custom problem, pass the problem instantiation to the environment instantiation. The rest of evlauation will be untouched since changing the problem is simply making sure your attacker is being called and tokenized correctly during attacker-target evaluation rollouts. 

See the [quick_start_evaluation](../quick_start_evaluation.md) guide for more information on the evaluation steps. 

```python
def main():
    DEVICE = "cuda"

    # Load evaluation prompts
    with open("prompts_reddit_test.json") as f:
        PROMPTS = json.load(f)

    # Instantiate problem
    problem = GPT2EvaluationProblem(ATTACKER_MODEL, DEVICE, LlamaGuardModerator())

    # Create environment & evaluator
    env = ASTEnvironment(problem, PROMPTS, tree_width=1, tree_depth=3)
    evaluator = ASTEvaluator(env, seeds=PROMPTS)

    # Run evaluation
    metrics = evaluator.evaluate(n_rollouts=20, progress=True)

    # Save results
    evaluator.write_json(metrics, "metrics.json")


if __name__ == "__main__":
    main()
```

---

## Recap

* **Most users don't need to write new classes** — just pick `HFEvaluationProblem` (non-GPT2) or `GPT2EvaluationProblem`.
* If you already subclassed a problem for training, reuse it for evaluation and only swap in your trained attacker.
* For fully custom architectures, subclass `ASTProblem` directly. Visit the [Problem Customization guide](../customizing_training/problems.md) for more information on the Problem class.
