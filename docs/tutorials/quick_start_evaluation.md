# How to Run an Evaluation / Red Team

RL-based red-teaming uses reinforcement learning to train an **attacker** that generates test cases likely to elicit unsafe outputs from a **target** model. This tutorial shows how to run evaluations (red-team rollouts) using a **pre-trained attacker** against a target model.

!!! note "Prerequisite"
This guide assumes you already trained an attacker (see **Quick Start: Training**). You’ll point evaluation at that saved attacker checkpoint.

---

## Quick Start

Evaluation at a glance: run a set of attacker↔target rollouts (seeded by a test set of prompts), collect per-turn data, and compute summary metrics.

### 1) Setup: imports, model paths, and device

Load dependencies and define the Hugging Face–style models you’ll use for **attacker** and **target**.

```python
from typing import List
from statistics import mean
import json
import torch
from transformers import AutoTokenizer  # optional if you post-process text

from astra_rl.ext.transformers import HFASTProblem
from astra_rl import ASTEnvironment, DetoxifyModerator
# Optional (richer eval nodes). If available in your build:
from astra_rl.methods.ast_problem import CustomNode  # optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths or HF model IDs
ATTACKER_MODEL_PATH = "path/or/hub-id/of/attacker"  # e.g., "/home/checkpoints/best" or "gpt2"
TARGET_MODEL_PATH   = "path/or/hub-id/of/target"    # e.g., "meta-llama/Llama-3-8B" (example)
```

### 2) Load evaluation prompts

Evaluation prompts **start** the attacker–target conversation. Make sure they:

1. Match the context you care about (e.g., diagnostic agent prompts for a diagnostic system).
2. Were **never** seen during training or dev.
3. Are provided as a **list of strings**.

```python
with open("prompts_reddit_test.json") as f:
    PROMPTS = json.load(f)  # e.g., ["prompt 1", "prompt 2", ...]
```

### 3) Instantiate the Problem and Environment

Use the preconfigured **HFASTProblem** (HF-friendly) and the **ASTEnvironment** (ASTPrompter-style rollouts).

```python
problem = HFASTProblem(
    attacker_model_id=ATTACKER_MODEL_PATH,
    target_model_id=TARGET_MODEL_PATH,
    baseline_model_id=TARGET_MODEL_PATH,  # a simple default; can also use a frozen attacker ref
    moderator=DetoxifyModerator(),
    device=DEVICE,
)

env = ASTEnvironment(
    problem=problem,
    prompts=PROMPTS,
    tree_width=1,  # single-path evaluation
    tree_depth=3,
)
```

### 4) Run evaluation rollouts and collect metrics

The snippet below performs a **single-path** rollout per prompt (depth = 3, width = 1) and demonstrates how to aggregate a few simple metrics. Use `env.final_reward(graph)` to read the reward at the leaf of the path.

```python
rows: List[dict] = []
summaries = []

for prompt in PROMPTS:
    # Single-path evaluation
    g = env.eval_rollout(prompt)

    # Final reward at the leaf (width=1)
    leaf_reward = env.final_reward(g) or 0.0

    # Minimal transcript rows (per-turn)
    # Each node is a (context, attack, response, reward, children) tuple.
    path_nodes = g.children[0].children if g.children else []  # depth >= 2
    rows.extend({
        "prompt": prompt,
        "attack": n.attack,
        "response": n.response,
        "reward": float(n.reward),
    } for n in path_nodes)

    # Optional: richer metrics if CustomNode is enabled in your build
    # (e.g., attack_tox, target_tox, attack_logprob)
    try:
        avg_attack_tox = mean([getattr(n, "attack_tox", 0.0) for n in path_nodes]) if path_nodes else 0.0
        avg_target_tox = mean([getattr(n, "target_tox", 0.0) for n in path_nodes]) if path_nodes else 0.0
        avg_attack_lp  = mean([getattr(n, "attack_logprob", 0.0) for n in path_nodes]) if path_nodes else 0.0
    except Exception:
        avg_attack_tox = avg_target_tox = avg_attack_lp = 0.0

    summaries.append({
        "prompt": prompt,
        "final_reward": float(leaf_reward),
        "avg_attack_tox": float(avg_attack_tox),
        "avg_target_tox": float(avg_target_tox),
        "avg_attack_logprob": float(avg_attack_lp),
    })

# At this point, `rows` holds per-turn data and `summaries` holds per-prompt aggregates.
# You can dump to JSON/CSV for analysis or visualization.
```

??? info "Prefer richer node metrics out-of-the-box?"
If your build includes `env.handle_eval_prompt(...)` that returns `CustomNode` entries (with toxicity and logprob fields), you can call it for each prompt and skip the `getattr(...)` checks above.

---
