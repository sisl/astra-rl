# Evaluation Systems

When performing an **evaluation**, you need to create a *system class* that correctly loads your **trained tester model** and its tokenizer.

Most of the time, the only change required is **how the tester model and tokenizer are instantiated**. All rollout logic and evaluation APIs remain the same.

---

## 1. Using Hugging Face Models (non-GPT2)

If your trained tester is a Hugging Face model that does **not** have a fixed maximum context length (e.g. LLaMA-3), you can simply use `HFEvaluationSystem`.

```python
from astra_rl.ext.transformers import HFEvaluationSystem

system = HFEvaluationSystem(
    tester_model="/path/to/your/tester/checkpoint",
    tester_base_model_id="meta-llama/Meta-Llama-3-8B",  # base tokenizer
    target_model_id="meta-llama/Meta-Llama-3-8B",
    device="cuda"
)
```

!!! tip
    See the [Quick Start: Evaluation](../quick_start_evaluation.md) tutorial or the [full HF evaluation example](https://github.com/sisl/astra-rl/tree/main/examples/ast_huggingface_eval.py).

---

## 2. Using GPT-2–based Testers

GPT-2 has some quirks (fixed max length of 1024, special padding setup).
We provide [`GPT2EvaluationSystem`](https://github.com/sisl/astra-rl/tree/main/examples/ast_gpt2_eval.py), which handles this automatically:

```python
from ast_gpt2_eval import GPT2EvaluationSystem

system = GPT2EvaluationSystem(
    tester_model="/path/to/tester/checkpoint",
    device="cuda"
)
```

**Key details:**

* Inherits from `GPT2DetoxifySystem`.
* Only overrides `__init__` to let you pass in a custom tester and scorer.
* Assumes:

  * Target = `"gpt2"`
  * Tester = GPT-2–based adversarial model.

!!! tip
    See the [full GPT2 evaluation example](https://github.com/sisl/astra-rl/blob/main/examples/ast_gpt2_eval.py).
---

## 3. Fully Custom Testers or Targets

If you are using a **completely custom pre-trained tester or target**, you will need to define your own subclass of `ASTSystem`.
This subclass must:

1. Instantiate tester, target, and tokenizers.
2. Implement rollout logic (text generation given context).

See the [System Customization guide](../customizing_training/problems.md) for details.

!!! note
    If you already created a custom system class for **training**, it is often easiest to **subclass it for evaluation** and just modify the tester instantiation.

    For example, the [`GPT2EvaluationSystem`](https://github.com/sisl/astra-rl/blob/main/examples/ast_gpt2_eval.py) is a thin subclass that changes only the constructor.

---

## 4. Example: GPT-2 Custom Evaluation System

Here's a concrete example showing how to create a custom system that loads a trained GPT-2 tester and a standard GPT-2 target:

```python

TESTER_MODEL = "./checkpoints/gpt2/best"

class GPT2EvaluationSystem(GPT2DetoxifySystem):
    """
    Same API/behavior as GPT2DetoxifySystem, but with a custom tester and scorer.
    Assumes target is GPT-2.
    """

    def __init__(self, tester_model: str = TESTER_MODEL,
                 device: str = "cpu",
                 scorer: Optional[DetoxifyScorer] = None):
        ASTSystem.__init__(self, scorer or DetoxifyScorer())
        self.device = device

        # Tester (trained GPT-2 adversary)
        self.tester = AutoModelForCausalLM.from_pretrained(tester_model).to(device)

        # Target (plain GPT-2)
        self.target = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        # Shared tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.tester.config.pad_token_id = self.tokenizer.eos_token_id
        self.target.config.pad_token_id = self.tokenizer.eos_token_id

        # Max context length
        self.max_ctx = int(getattr(self.tester.config, "n_positions",
                                   getattr(self.tester.config, "max_position_embeddings", 1024)))
```

---

## 5. Putting It All Together

After creating your custom system, pass the system instantiation to the sampler instantiation. The rest of evaluation will be untouched since changing the system is simply making sure your tester is being called and tokenized correctly during tester-target evaluation rollouts.

See the [quick_start_evaluation](../quick_start_evaluation.md) guide for more information on the evaluation steps.

```python
def main():
    DEVICE = "cuda"

    # Load evaluation prompts
    with open("prompts_reddit_test.json") as f:
        PROMPTS = json.load(f)

    # Instantiate system
    system = GPT2EvaluationSystem(TESTER_MODEL, DEVICE, LlamaGuardScorer())

    # Create sampler & evaluator
    sampler = ASTSampler(system, PROMPTS, tree_width=1, tree_depth=3)
    evaluator = ASTEvaluator(sampler, seeds=PROMPTS)

    # Run evaluation
    metrics = evaluator.evaluate(n_rollouts=20, progress=True)

    # Save results
    evaluator.write_json(metrics, "metrics.json")


if __name__ == "__main__":
    main()
```

---

## Recap

* **Most users don't need to write new classes** — just pick `HFEvaluationSystem` (non-GPT2) or `GPT2EvaluationSystem`.
* If you already subclassed a system for training, reuse it for evaluation and only swap in your trained auditor.
* For fully custom architectures, subclass `ASTSystem` directly. Visit the [System Customization guide](../customizing_training/problems.md) for more information on the System class.
