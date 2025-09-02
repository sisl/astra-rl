# How to create a custom **TrainingConfiguration** and **Trainer** (ASTRA-RL)

## What the `Trainer` does

`astra_rl/training/trainer.py` (the base Trainer class) is responsible for:

1. Instantiating the **optimizer** that updates the attacker’s policy to reduce training loss.
2. Instantiating the **training harness** (handles online data collection; you normally don’t edit this).
3. Running the **main training loop** in `train()` using your config hyperparameters.
   **Note:** the base loop **does not** perform evaluation, model saving, or logging.

> The harness calls your **Algorithm** on each batch and returns `(loss, step_logs)`. The base `Trainer` uses the loss for optimization and **discards** `step_logs`. Use 'HFASTTrainer' or override `train()` if you want logging/eval/checkpointing.

## TL;DR decision guide

* **Just train, no eval/checkpointing?** Use the base `Trainer` (Sections 1 & 2).
* **Want periodic dev eval + HF checkpoints?** Use `HFASTTrainer` / `HFASTConfiguration` (Sections 1 & 3).
* **Need a custom loop/scheduler/saving policy?** Subclass `Trainer` (Section 4).

---

## 1) Define your TrainingConfiguration (hyperparameters)

Instantiate `TrainingConfiguration` with the knobs you care about. These values drive how the base trainer is initialized.

```python
from astra_rl import Trainer, TrainingConfiguration

config = TrainingConfiguration(
    lr=1e-5,
    batch_size=4,
    optimizer="adamw",                 # ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
    gradient_accumulation_steps=1,     # optimizer.step() every N backwards
    training_steps=1000,               # number of calls to experience()
    num_episodes_per_experience=2,     # rollouts per experience() call
)
```

> **Note:** `training_steps` = number of times we call `experience()`;
> **Total rollouts ≈** `training_steps × num_episodes_per_experience`.

---

## 2) Fast path A — use the base `Trainer` as-is

If you’re happy with a simple loop that **optimizes** on the algorithm loss and **does not** eval, save, or log:

```python
trainer = Trainer(config=config, environment=env, algorithm=solver)
trainer.train()  # base loop; no eval/checkpointing/logging
```

To save the final model afterward:

```python
problem.attacker.save_pretrained("final_ckpt")
problem.tokenizer.save_pretrained("final_ckpt")
```

---

## 3) Fast path B — use the HF-compatible trainer with periodic eval & saving

Preconfigured subclass that runs periodic dev evaluation and saves Hugging Face checkpoints.

```python
from astra_rl.ext.transformers.hf_ast_problem import (
    HFASTTrainer,
    HFASTConfiguration,
)

config  = HFASTConfiguration()  # or use your own TrainingConfiguration
trainer = HFASTTrainer(
    config=config,
    environment=env,
    algorithm=solver,
    dev_prompts=DEV_PROMPTS,   # iterable of prompts
    eval_every=100,            # run dev eval every N steps
    ckpt_dir="checkpoints",    # where to save HF checkpoints
)
trainer.train()
```

---

## 4) Fully custom — write your own `Trainer` subclass

Use this when you need **custom evaluation cadence**, **checkpointing**, **logging**, **LR scheduling**, **early stopping**, etc. You **may** override `train()`; if you don’t, you inherit the base implementation.

```python
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

    def __init__(self, config: MyConfig, environment, algorithm, dev_prompts=None):
        super().__init__(config, environment, algorithm)
        self.dev_prompts = dev_prompts or []
        os.makedirs(self.config.ckpt_dir, exist_ok=True)

    # optional but encouraged
    def _save_hf(self, step: int) -> None:
        """Save your attacker and its tokenizer"""

    # optional method
    @torch.no_grad()
    def _eval_dev(self, step: int, tag: str = "dev"):
        """Run a lightweight evaluation on dev prompts. Fill in your logic."""
        pass

    # required method
    def train(self):
        """Implement your custom training loop!"""
        # see astra_rl.ext.transformers.hf_ast_problem for an implemented custom class example
        pass        
```

---

## Notes & best practices

* **Don’t modify the Harness** unless you’re sure you can't change data-collection semantics though changes to the trainer/config/environment/algorithm.
* **Logging:** only log scalars (detach tensors) to avoid retaining graphs: `logs["loss"] = float(loss.detach().item())`.
* **Checkpoints:** saving the attacker/tokenizer is usually sufficient; if your algorithm maintains extra state, save that too.
* **Imports:** to use the HF trainer directly,
  `from astra_rl.ext.transformers.hf_ast_problem import HFASTTrainer, HFASTConfiguration`.
