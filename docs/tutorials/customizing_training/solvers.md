# Solvers (RL Algorithm) 

**Solvers** (a.k.a. algorithms) define *how learning happens*. They consume rollout graphs from the **Sampler**, ask the **System** for model log-probs/rewards, and return a scalar **loss** (plus optional logs) to the **Trainer**. In ASTRA-RL a solver subclasses `Algorithm[...]` and typically implements three things:

1. `flatten(graph)` → turn a rollout `Graph` into per-sample **Steps**
2. `collate_fn(steps)` → batch those steps into a **Batch**
3. `step(batch)` → compute the training **loss** and a `logs` dict

---

## 1. What Solvers Do

Given rollouts (graphs of auditor–target turns), a solver decides **what examples to learn from** (via `flatten`), **how to batch them** (`collate_fn`), and **what objective to optimize** (`step`). This keeps "*how we learn*" separate from:

* **Sampler**: *how data is collected/structured* (single path vs tree, etc.)
* **System**: *how models are run* (log-probs, rewards, advance logic)

---

## 2. Built-in Solvers/Examples

ASTRA-RL includes preference-learning solvers commonly used for LM alignment/red-teaming:

* **DPO** — Direct Preference Optimization (pairwise preferred vs rejected)
* **IPO** — Implicit Preference Optimization (margin-style objective over log-ratio differences)
* **PPO** - Proximal Policy Optimization 

These serve as concrete references for writing your own solver. Find the code for these solvers [here](https://github.com/sisl/astra-rl/blob/main/src/astra_rl/core/algorithm.py)!

---

## 3. Ways to Customize

### 3.1 Fast path: adapt a built-in (e.g., DPO → IPO)

If your rollout selection and batching are the same, you can **reuse `flatten` and `collate_fn`** and only change the **loss** in `step`. IPO in our codebase demonstrates this pattern by inheriting from DPO and overriding `step`. Therefore, if you are only making a small change to how the loss is calculated, a great option would be to inheret from the DPO, IPO or PPO and ovverid 'step' to include your custom loss calculation.

### 3.2 Full control: subclass `Algorithm`

When your algorithm needs a different sampling strategy, subclass `Algorithm[...]` and implement `flatten`, `collate_fn`, and `step` to match your data/learning objective.

---

## 4. Required Interface

### 4.1 Step/Batch data contracts

Define explicit dataclasses that encode exactly what your algorithm needs.

```python
from dataclasses import dataclass
from typing import Generic, Sequence
from astra_rl.core.common import StateT, ActionT

@dataclass
class MyStep(Generic[StateT, ActionT]):
    context: StateT
    action: ActionT
    reward: float  # or advantage/return/log-ratio/etc.

@dataclass
class MyBatch(Generic[StateT, ActionT]):
    contexts: Sequence[StateT]
    actions:  Sequence[ActionT]
    rewards:  torch.Tensor  # tensors for math
```

> Keep these minimal and algorithm-specific. They are the *contract* between your data selection (`flatten`) and your loss (`step`).

### 4.2 `flatten`, `collate_fn`, `step` contracts

* **`flatten(graph: Graph) -> Sequence[Step]`**
  Select and transform nodes/edges from the rollout graph into per-sample `Step`s (BFS/DFS as you like).
* **`collate_fn(steps: Sequence[Step]) -> Batch`**
  Convert a list of steps into batched tensors/sequences for efficient training.
* **`step(batch: Batch) -> tuple[torch.Tensor, dict]`**
  Compute a scalar **loss** (used for backprop) and a **logs** dict of floats (the base trainer may ignore them; custom trainers can log them).

### 4.3 Interacting with `System`

Your solver calls into the `System` for model computations:

* `system._get_auditor_logprobs_and_validate(contexts, actions)`
* `system._get_baseline_logprobs_and_validate(contexts, actions)`
* optionally: `system.get_target_logprobs(...)`, `system.reward(...)`, etc.

**Tip:** Target/baseline log-prob calls usually should be in `torch.no_grad()`; the auditor's log-probs must require grad.

---

## 5. Best Practices & Sanity Checks

* **Pairwise methods need width ≥ 2.** For DPO/IPO, set `tree_width >= 2` so each context has at least two candidate actions.
* **Stable scales.** Keep losses well-scaled (e.g., use a `beta` like in DPO/IPO). Normalize or clip rewards if needed.
* **Efficient batching.** Vectorize log-prob calls; avoid per-item model runs.
* **Validate shapes.** Collated tensors must be aligned and same length.
* **Freeze the ref/baseline.** Only attacker params should receive gradients.
* **KL anchor (when applicable).** If training drifts, increase KL pressure (or use adaptive control) where appropriate.

---

## 6. Plug into the Trainer

Instantiate and pass your solver to the trainer:

```python
solver  = DPO(system, beta=0.1)  # or IPO(...)
trainer = Trainer(config=config, sampler=sampler, algorithm=solver)
trainer.train()
```

Under the hood, the **Trainer** will:

1. collect rollout graphs,
2. call your solver's `flatten` to produce `Steps`,
3. use your solver's `collate_fn` to form batches, and
4. call your solver's `step` to get `(loss, logs)`.

> The base `Trainer` uses `loss` for optimization and may ignore `logs`. Use `HFASTTrainer` or a custom trainer to evaluate and checkpoint.

---

## 7. Debug Checklist

* **Shapes match:** `len(prefixes) == len(pos) == len(neg)` (or analogous fields).
* **Gradients only through attacker:** wrap baseline/target log-prob calls in `torch.no_grad()` if you surface them directly.
* **Finite values:** check for `nan/inf` in losses and rewards (clip/normalize if necessary).
* **Tree width OK:** preference solvers require `tree_width ≥ 2`.
* **KL anchor:** if the auditor drifts, increase β or add an explicit KL penalty to the loss.
* **Determinism:** set seeds and/or make selection in `flatten` deterministic to repro bugs.

---
