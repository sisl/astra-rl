# How to Create a Custom Solver (RL Algorithm)

In ASTRA-RL, a **solver** subclasses `Algorithm[...]` and implements three things:

1. **`flatten(graph)`** – turn a rollout `Graph` into a list of per-sample **Steps** your solver trains on
2. **`collate_fn(steps)`** – batch those steps into a **Batch** the harness can load via a DataLoader
3. **`step(batch)`** – compute the training **loss** (and log metrics), returning `(loss, logs)`

This isolates “how we learn” (the solver) from “how data is collected” (the Environment) and “how models run/interact” (the Problem).

---

## Choose your data contract

First, define your own `@dataclass Step`/`Batch` to match your algorithm. These dataclasses determine what information is stored at each step 
of learning and hold the information across a total batch. (TODO: improve)

---

## example

```python
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Sequence, Tuple
import torch

from astra_rl.core.algorithm import Algorithm
from astra_rl.core.common import StateT, ActionT
from astra_rl.core.environment import Graph

# 1) Define your per-sample record
@dataclass
class MyStep(Generic[StateT, ActionT]):
    context: StateT
    action: ActionT
    reward: float  # or returns/advantages/etc.

# 2) Define your batch
@dataclass
class MyBatch(Generic[StateT, ActionT]):
    contexts: Sequence[StateT]
    actions: Sequence[ActionT]
    rewards: torch.Tensor  # torch for math
```
## Plugging your solver into the harness

To integrate the solver, simply instantiate in your main code and give it to the harness as a parameter. Your solver
should provide the harness with a list of steps. 

```python
solver = MyAlgo(problem, kl_coeff=0.02)
harness = Harness(
    env,
    solver,
    num_episodes_per_experience=2,
    dataloader_kwargs={"batch_size": 8, "shuffle": True},
)
for step in range(1000):
    for batch in harness.experience():     # yields MyBatch via your collate_fn
        loss, logs = harness.step(batch)   # calls solver.step(...)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        logs["step"] = step
        harness.log_current_step(logs)
```

---

## Debug checklist

* **Shapes**: `flatten` produces a list; `collate_fn` returns tensors/seqs with consistent lengths.
* **Gradient flow**: only `get_attacker_logprobs` should require grad; keep target/baseline calls in `torch.no_grad()`.
* **Rewards**: are finite and roughly bounded; consider normalization or clipping.
* **KL anchor**: if training diverges, raise `kl_coeff` (or use adaptive control).
* **Tree width**: preference algorithms need `tree_width ≥ 2`; reward methods are simplest with `tree_width = 1`.
* **Determinism**: seed env rollouts to reproduce a bug.

---

## Adapting to your data

* Per-turn rewards? Put them into `MyStep` and compute **returns** (discounted sum) in `flatten` or `collate_fn`.
* Off-policy data? Add `old_logp` to `MyStep` and implement importance sampling/clip.
* Extra features (toxicity, length, success flags)? Add fields to `MyStep` and use them in the loss.

---
