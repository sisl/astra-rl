# Trainers

**Trainers** run the optimization loop that updates your auditor. They wire together the **sampler** (rollout collection), the **algorithm/solver** (computes a loss from rollouts), and the **optimizer** (updates model weights). In ASTRA-RL you can use a minimal, no-frills base trainer or a preconfigured, Hugging Face–friendly trainer that handles evaluation and checkpointing.

This guide explains what a trainer does, what ASTRA-RL ships with, and how to implement or customize your own trainer and training configuration.

---

## 1. What Trainers Do

The base trainer in `astra_rl/training/trainer.py` is responsible for:

1. **Optimizer setup** — creates the optimizer that updates the auditor's weights.
2. **Harness orchestration** — uses the **Harness** to collect rollouts and feed batches to your **Algorithm** (solver).
3. **Main training loop** — calls `train()` to iterate for `training_steps`.

**Important:** the base loop is intentionally minimal. It **does not** perform evaluation, checkpointing, or external logging.

!!! note 

    The Harness invokes your **Algorithm** on each batch and returns `(loss, step_logs)`. The base `Trainer` uses the loss for backprop and **discards** `step_logs`. Use `HFASTTrainer` or subclass `Trainer` if you need logging/eval/checkpointing.

---

## 2. Built-in Trainers

* **`Trainer` (base)** — minimal loop: collect → compute loss → optimize. No eval/saving/logging.
* **`HFASTTrainer`** — Hugging Face–friendly trainer that performs **periodic dev evaluation** and **saves HF checkpoints**, driven by `HFASTConfiguration` (or your own config).

---

## 3. Ways to Customize

### 3.1 Fast path: use the base `Trainer`

If you just want a lean optimization loop (no eval/checkpointing/logging), use `Trainer` with a `TrainingConfiguration`. See [6.1](#61-minimal-usage-of-the-base-trainer).

### 3.2 Fast path: HF-compatible trainer (`HFASTTrainer`)

If you want periodic dev evaluation and automatic Hugging Face checkpointing, use `HFASTTrainer` with `HFASTConfiguration`. See [6.2](#62-periodic-eval-hf-checkpoints-via-hfasttrainer).

### 3.3 Full control: subclass `Trainer`

Need custom evaluation cadence, model-saving policy, learning-rate schedules, gradient accumulation, early stopping, or logging destinations (e.g., Weights & Biases)? Subclass `Trainer` and override `train()` (and optional helpers). See [6.3](#63-write-a-custom-trainer-with-eval-saving-and-grad-accumulation) and [6.4](#64-early-stopping-or-custom-lr-schedules).

---

## 4. Required Interface

### 4.1 `TrainingConfiguration` knobs

Instantiate `TrainingConfiguration` with the hyperparameters you care about. These values drive how the trainer and optimizer are initialized.

```python
from astra_rl import TrainingConfiguration

config = TrainingConfiguration(
    lr=1e-5,
    batch_size=4,
    optimizer="adamw",                 # one of ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
    gradient_accumulation_steps=1,     # call optimizer.step() every N backward passes
    training_steps=1000,               # number of experience() calls
    num_episodes_per_experience=2,     # rollouts sampled per experience() call
)
```

* **`training_steps`** = number of Harness `experience()` iterations.
* **Approx. total rollouts** ≈ `training_steps × num_episodes_per_experience`.

### 4.2 What the Trainer expects from the Harness/Algorithm

* The **Harness** returns an iterable/batches of experiences.
* For each batch, the trainer calls the **Algorithm** (solver) via the Harness to process the experience and obtain:

  ```python
  loss, step_logs = harness.step(batch)
  ```

  * `loss`: a scalar tensor used for backprop.
  * `step_logs`: optional dict of scalars (e.g., reward stats). The **base** trainer ignores these; custom trainers can log them.

---

## 5. Best Practices & Sanity Checks

* **Don’t hack the Harness** unless you truly need different data-collection semantics; most customization belongs in the trainer/config/environment/algorithm.
* **Detach when logging:** `logs["loss"] = float(loss.detach().item())` to avoid holding computation graphs.
* **Checkpoint sensibly:** attacker + tokenizer is usually enough; if your algorithm has extra state, save it too.
* **Batching vs. accumulation:** prefer reasonable batch sizes; use `gradient_accumulation_steps` when memory is tight.
* **Reproducibility:** seed PyTorch/NumPy and pass a `seed` through your environment when possible.
* **Validation cadence:** dev eval can be slow—choose `eval_every` that matches your budget.

---

## 6. How-Tos

### 6.1 Minimal usage of the base `Trainer`

A simple loop that optimizes the algorithm loss with **no** eval, checkpoints, or logging:

```python
from astra_rl import Trainer

trainer = Trainer(
    config=config,
    sampler=sampler,
    algorithm=solver,
)
trainer.train()

# Optionally save the final HF model/tokenizer:
system.auditor.save_pretrained("final_ckpt")
system.tokenizer.save_pretrained("final_ckpt")
```

---

### 6.2 Periodic eval + HF checkpoints via `HFASTTrainer`

Use the preconfigured, Hugging Face–compatible trainer for periodic dev evaluation and automatic checkpointing.

```python
from astra_rl.ext.transformers.hf_ast_problem import (
    HFASTTrainer,
    HFASTConfiguration,
)

config  = HFASTConfiguration()  # or your own TrainingConfiguration
trainer = HFASTTrainer(
    config=config,
    sampler=sampler,
    algorithm=solver,
    dev_prompts=DEV_PROMPTS,   # iterable of prompts for evaluation
    eval_every=100,            # run dev eval every N steps
    ckpt_dir="checkpoints",    # HF-format checkpoints saved here
)
trainer.train()
```

---

### 6.3 Write a custom trainer with eval, saving, and grad accumulation

Subclass `Trainer` to add evaluation cadence, HF-style saving, gradient accumulation, and logging.

<pre style="font-size:0.8rem; line-height:1.35;"><code class="language-python">
import os
import torch
from astra_rl import Trainer, TrainingConfiguration
from astra_rl.logging import logger

class MyConfig(TrainingConfiguration):
    def __init__(self):
        super().__init__(
            lr=1e-5,
            batch_size=4,
            optimizer="adamw",
            gradient_accumulation_steps=1,
            training_steps=1000,
            num_episodes_per_experience=2,
        )
        # Custom fields for your subclass:
        self.eval_every = 100
        self.ckpt_dir = "checkpoints"

class MyTrainer(Trainer):
    """
    Extends the base trainer with:
      - periodic dev-set evaluation
      - HF-format checkpointing
      - optional grad accumulation
    """

    def __init__(self, config: MyConfig, sampler, algorithm, dev_prompts=None):
        super().__init__(config, sampler, algorithm)
        self.dev_prompts = dev_prompts or []
        os.makedirs(self.config.ckpt_dir, exist_ok=True)

    # optional but encouraged
    def _save_hf(self, step: int) -> None:
        """Save your auditor and its tokenizer"""

    # optional method
    @torch.no_grad()
    def _eval_dev(self, step: int, tag: str = "dev"):
        """Run a lightweight evaluation on dev prompts. Fill in your logic."""
        pass

    # required method
    def train(self):
        """Implement your custom training loop!"""
        # see astra_rl.ext.transformers.hf_ast_system for an implemented custom class example
        pass       
</code></pre>

---

### 6.4 Early stopping or custom LR schedules

Inside your custom `train()`:

* **Early stopping:** track a validation metric (e.g., dev score from `_eval_dev`) and stop after `x` steps with no improvement.
* **LR scheduling:** instantiate a PyTorch scheduler after the optimizer and call `scheduler.step()` each iteration or epoch; store scheduler state alongside checkpoints if you need exact resumption.

---

## 7. Full Examples

* **Custom AST system **with** trainer:** `examples/GPT2_v_GPT2/ast_trainer.py`
* **Source for HF-compatible trainer/config:** `astra_rl/ext/transformers/hf_ast_system.py`

Use these as references when writing up your own training loop or extending the provided trainers.
