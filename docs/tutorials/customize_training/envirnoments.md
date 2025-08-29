# How to Customize the Environment

The **environment** defines how attacker–target interactions are produced and packaged for training. It controls:

* how rollouts are generated (single path vs. tree),
* how states advance across turns,
* what per-step data (actions, responses, rewards) is stored,
* and what the solver receives as input.

ASTRA-RL ships with `ASTEnvironment`, which mirrors the rollout structure used in **ASTPrompter**. You can subclass it—or the base `Environment`—to support:

* multi-agent or multi-turn conversations,
* tree-structured or flat trajectories,
* custom state-advance logic,
* alternative reward shaping or logging.

---

## Option A: Subclass the base `Environment`

```python
from astra_rl.core.environment import Environment

class MyCustomEnvironment(Environment[str, str]):
    ...
```

The base `Environment` defines the interface the training harness expects. Make sure you implement the mandatory rollout() method.

## Option B: Subclass an existing environment

If your rollout logic is “like AST but with a twist,” subclass `ASTEnvironment` and override only the parts you need (e.g., the expansion logic, evaluation, or metrics extraction).

```python
from astra_rl import ASTProblem, ASTEnvironment 
from astra_rl.core.environment import Node, Graph

class MyASTVariant(ASTEnvironment):
    """
    Subclass of ASTEnvironment that performs your custom rollout logic
    """

    def __init__(...):
        super().__init__(problem, prompts, tree_width, tree_depth)
```

---

## Required: implement `rollout(...)`

Your environment **must** implement:

```python
def rollout(self, seed: Optional[int] = None) -> Graph[str, str]:
    ...
```

This function performs one training rollout and returns a `Graph` made of `Node`s:

* `Graph(context: str, children: List[Node])`
* `Node(context: str, attack: str, response: str, reward: float, children: List[Node])`

At a minimum, each node should capture:

* `context` (state so far, e.g., conversation text),
* `attack` (attacker’s utterance / action),
* `response` (target/defender’s reply),
* `reward` (float),
* `children` (next steps; empty for leaf nodes).

> **Tip:** Make sure your rollout matches your solver’s data contract.
    Preference-based methods (e.g., DPO/IPO/ORPO) need preference pairs, so the easiest online setup is tree_width ≥ 2 to sample two candidates per prompt. Reward-based policy-gradient methods (e.g., PPO/A2C) do not require pairs; tree_width = 1 is typical.

### Useful `ASTProblem` helpers inside your environment

These are the public, batch-friendly methods you’ll typically call during rollout:

* `rollout_prompt_with_attacker(prompts: Sequence[str]) -> Sequence[str]`
* `rollout_prompt_with_target(prompts: Sequence[str]) -> Sequence[str]`
* `reward(prompts, attacks, responses) -> Sequence[float]`
* `advance(prompt, attack, response) -> str`  *(returns the next state)*

> **Tip:** These APIs are vectorized—**always pass lists** (even if length 1) for best performance and simpler code.

---

## (Optional) Evaluation metrics

If you want per-turn metrics like toxicity or likelihood, add a helper that **walks a single-path graph** and computes metrics at each turn. These helper functions can be useful when evaluating models after training. See the quick_start_evaluation guide for more information. 

> **Tip:** Use the moderator in **batch mode**: `moderator.moderate(list_of_texts)`.

---

## Best practices & common pitfalls

* **Prefer batch calls.** The `ASTProblem` helpers are vectorized—avoid per-item model calls inside tight loops.
* **Determinism:** Thread a `seed` through `rollout()` for reproducible experiments.
* **Depth/width off-by-one:** Depth `= 0` should produce **no** nodes; verify leaf counts match expectations.
* **State advancement:** Always use `advance(state, attack, response)` to build the next state, even if it’s simple string concatenation—this keeps things consistent and testable.
* **Graph sanity checks:** Before training, print or visualize one `rollout()` to confirm the structure matches your mental model (fan-out, depth, rewards present).
* **Context length:** If your `Problem` uses LMs with max context, ensure your `Problem` handles truncation. Environments should not assume a specific token limit.
* **Evaluation vs. training:** Keep evaluation single-path (`width=1`) to simplify metrics and avoid combinatorial explosion.

---

## Quick validation checklist

1. `env.rollout()` returns a `Graph` with the expected shape (root + child nodes).
2. Every `Node` has `context`, `attack`, `response`, `reward` populated.
3. `advance()` is called exactly once per (context, attack, response).
4. Running two `rollout(seed=123)` calls yields the same graph.
5. `final_reward()` returns the last-step reward in a single-path rollout.
6. (If added) `eval_rollout(prompt)` respects `width=1` and your metrics extractor runs without per-item model calls.

---

## Example: printing a small rollout

```python
g = env.rollout(seed=7)
print("ROOT:", g.context)
for i, n in enumerate(g.children):
    print(f"[{i}] atk={n.attack!r} rsp={n.response!r} rew={n.reward:.3f} #children={len(n.children)}")
```

---

## Extending the data model

Need extra metadata (e.g., per-turn safety tags, KL terms, or timestamps)?
Subclass `Node` and/or wrap `Graph` with your own dataclasses, then adapt your solver to read those fields.
