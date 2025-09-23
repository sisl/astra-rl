# How to Run an Evaluation / Red Team

RL-based red-teaming uses reinforcement learning to train an **attacker** that generates test cases likely to elicit unsafe outputs from a **target** model. This tutorial shows how to run evaluations (red-team rollouts) using a **pre-trained attacker** against a target model.

!!! note "Prerequisite"
    This guide assumes you already trained an attacker (see **Quick Start: Training**). You’ll point evaluation at that saved attacker checkpoint.


!!! note "COMING SOON"
    Our team is currently working on the evaluation API to help you quickly evaluate your target model with a pre-trained RL-based attacker. Please check back soon!
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

### 4) Instantiate the Evaluator

... Coming Soon! 