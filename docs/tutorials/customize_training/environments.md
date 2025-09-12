# How to Customize the Environment

The **environment** defines how attacker–target interactions are generated and packaged for training and evaluation. It controls:

* how rollouts are generated (single path vs. tree),
* what per-step data (actions, responses, rewards) is stored,
* and what the solver receives as input.

!!! note 

    The environment also powers evaluation by running a single-path rollout (tree width = 1) and collecting metrics. See [Evaluation Quick Start](../quick_start_evaluation.md) for more details on evaluation.

ASTRA-RL ships with `ASTEnvironment`, which mirrors the rollout structure used in **ASTPrompter**. You can subclass it—or the base `Environment`—to support:

* multi-agent conversations,
* tree-structured or flat trajectories,
* custom state-advance logic,
* alternative reward shaping or logging.

---

## 1. Ways to Customize

### 1.1 Fast path: subclass `ASTEnvironment`

If your logic is “AST-like with a twist,” subclass `ASTEnvironment` and override only what you need (e.g., expansion logic, evaluation, or metrics extraction).

```python
from astra_rl import ASTProblem, ASTEnvironment
from astra_rl.core.environment import Node, Graph

class MyASTVariant(ASTEnvironment):
    """ASTEnvironment with custom rollout behavior."""

    def __init__(self, problem: ASTProblem, prompts, tree_width=2, tree_depth=3):
        super().__init__(problem, prompts, tree_width, tree_depth)

    # Example: override the recursive expansion
    # def __handle_prompt(self, prompt, depth, width=None):
    #     ...
```

### 1.2 Full control: subclass `Environment`

If you want to design the rollout structure from scratch, subclass `Environment`. You implement `rollout()` (and any helpers you like) and return a `Graph` of `Node`s that your solver expects.

```python
from typing import Optional
from astra_rl.core.environment import Environment, Node, Graph

class MyCustomEnvironment(Environment[str, str]):
    def rollout(self, seed: Optional[int] = None) -> Graph[str, str]:
        # Build and return a Graph made of Node[str, str]
        ...
```

> Use your `Problem`’s batch helpers for attacker/target calls and rewards. It keeps code concise and fast.

---

## 2. Required Interface

### 2.1 Nodes and Graphs

Your environment’s `rollout()` **must** return:

* `Graph(context: str, children: list[Node])`
* `Node(context: str, attack: str, response: str, reward: float, children: list[Node])`

At a minimum, each node should capture:

* `context` — state so far (e.g., conversation text),
* `attack` — attacker’s utterance / action,
* `response` — target/defender’s utterance / reply,
* `reward` — scalar float for this turn,
* `children` — next steps; empty for leaves.

> **Solver contract.** Make sure the structure and fields match what your solver consumes.
> • Preference-based methods (DPO/IPO/ORPO) usually want **pairs** → use `tree_width ≥ 2`.
> • Reward-based policy gradients (PPO/A2C) don’t need pairs → `tree_width = 1` is typical.

### 2.2 Helpful `ASTProblem` APIs

If you are using the 'ASTProblem' note these batch-friendly methods you’ll typically call during rollout:

* `rollout_prompt_with_attacker(prompts: Sequence[str]) -> Sequence[str]`
* `rollout_prompt_with_target(prompts: Sequence[str]) -> Sequence[str]`
* `reward(prompts, attacks, responses) -> Sequence[float]`
* `advance(prompt, attack, response) -> str` *(builds the next state)*

> These APIs are **vectorized**—pass lists (even length-1) for simplicity and speed.

---

## 3. Best Practices & Sanity Checks

* **Prefer batch calls.** Avoid per-item model calls in tight loops.
* **Determinism.** Thread a `seed` through `rollout()` for reproducibility.
* **Depth/width off-by-one.** Depth `0` should produce **no** nodes; verify expected leaf counts.
* **Always use `advance()`.** Even if it’s simple concatenation now, it future-proofs your code.
* **Context length.** Let the `Problem` handle truncation; the environment shouldn’t assume token limits.
* **Check your environment by printing a few rollouts.**

  ```python
  g = env.rollout(seed=7)
  print("ROOT:", g.context)
  for i, n in enumerate(g.children):
      print(f"[{i}] atk={n.attack!r} rsp={n.response!r} "
            f"rew={n.reward:.3f} children={len(n.children)}")
  ```

---

## 4. How-Tos

### 4.1 Create a custom Node or Graph

Need extra metadata (e.g., per-turn safety tags, KL terms, timestamps)? Subclass `Node` to take in additional data as shown below.

```python
from typing import List, Iterable, Optional, Tuple

# example custom node creation
class CustomNode(Node[str, str]):
    """
    A Node with extra per-turn metadata for evaluation/training diagnostics.
    Compatible anywhere a plain Node is expected (isinstance(CustomNode, Node) == True).
    """

    def __init__(
        self,
        context: str,
        attack: str,
        response: str,
        reward: float,
        children: Sequence["Node[str, str]"],
        *,
        attack_tox: float = 0.0,
        target_tox: float = 0.0,
        attack_logprob: float = 0.0,
    ):
        # Initialize the base Node fields first
        super().__init__(context, attack, response, reward, list(children))
        # Then attach your custom metrics
        self.attack_tox: float = float(attack_tox)
        self.target_tox: float = float(target_tox)
        self.attack_logprob: float = float(attack_logprob)
```

> Keep the original fields intact so existing solvers remain compatible.

### 4.2 Change rollout width/depth

Adjust at environment construction time:

```python
from astra_rl import ASTEnvironment

# 4 attacker continuations per context, depth of 2 attacker–defender turns
env = ASTEnvironment(problem, PROMPTS, tree_width=4, tree_depth=2)
```

> **Cost warning.** Width × depth increases compute and memory quickly—scale carefully.

### 4.3 Multi-agent conversations

To support multi-agent conversations, you will need to query all participating models when building the rollout. This will occur in your environment (likely in modifications to __handle_rollout or rollout) but it is largely up to you on how you want to style it. For example, one approach would be to subclasses `ASTEnvironment` and override the internal expansion to call **K** defenders, combine their responses, and reduce rewards. However, you can structure how you call the agents and how you want to model rewards to fit your needs. 

> If you maintain the same Node type (Node(context: str, attack: str, response: str, reward: float, children: List[Node])) and return a graph, you will be able to plug into available solvers. However, deviating from this structure may require you to create a custom solver to interpret the custom rollouts and calculate loss accordingly. 

> Training with multiple agents multiplies compute: each node now triggers more model calls and stores more data.
