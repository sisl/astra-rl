# How to Run an Evaluation

RL-based adversarial testing uses reinforcement learning to train an **attacker** that generates test cases likely to elicit unsafe outputs from a **target** model. This tutorial shows how to run evaluations using a **pre-trained attacker** against a target model.

!!! note "Prerequisite"
    This guide assumes you already trained a GPT2 attacker (see **Quick Start: Training**). You’ll point evaluation at that saved attacker checkpoint.

---

## Quick Start

Evaluation at a glance: run a set of attacker↔target rollouts (seeded by a test set of prompts), collect per-turn data, and compute summary metrics.

### 1) Setup: imports, model paths, and device

Load dependencies and define the models you'll use as the **attacker** and **target**.

```python
# import dependencies 
from typing import Optional
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from astra_rl import ASTProblem, DetoxifyModerator, ASTEnvironment
from ast_eval_test.py import EvaluationProblem
from astra_rl.core.evaluator import Evaluator

# import ASTEvaluator for the quickstart guide
from astra_rl.methods.ast_problem import ASTEvaluator

# optional: only import llamaguardModerator if you want to use it during evaluations
from astra_rl.moderators.llamaGuard import LlamaGuardModerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your pre-trained attacker gpt2 model
ATTACKER_MODEL = "path/or/hub-id/of/attacker"  # e.g., "/home/checkpoints/best" or "gpt2"
TARGET_MODEL   = "gpt2"    # e.g., "meta-llama/Llama-3-8B" (example)
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

Use the preconfigured **EvaluationProblem** (takes in a pre-trained GPT2 model) and the **ASTEnvironment** (ASTPrompter-style rollouts).

```python
# instantiate the evaluation problem with your attacker and target models
problem = EvaluationProblem(ATTACKER_MODEL, TARGET_MODEL, DEVICE, DetoxifyModerator())

# instantiate the AST environment - no adjustments needed because already has eval_rollout
env = ASTEnvironment(problem, PROMPTS, 1, 3)
```
If you would like to evaluate HF-based attackers, use the HFASTProblem instead of the EvaluationProblem. If you would like to evaluate a custom attacker, see the [Evaluation Problem Customization](customizing_evaluation/evaluation_problems.md) guide.

### 4) Evaluate!

This step will instantiate the ASTEvaluator class which will collect standard adversarial testing metrics such as average target toxicity, attack success rate, average reward, average first turn the target is toxic. 

Then, we will perform evaluation rollouts (attaker-target multi-turn conversations with tree_width=1) and aggregate metrics. The metrics will be saved to your repository as a json. 

```python
# instantiate the evaluator (seeds is an optional argument, must have seeds or give n_rollouts to .evaluate below)
evaluator = ASTEvaluator(env, seeds=PROMPTS)

# collect metrics by running n_rollouts
metrics = evaluator.evaluate(n_rollouts=20, progress=True)

# save metrics to json file
Evaluator.write_json(metrics, "metrics.json")
```

The source code for ASTEvaluator is located at examples/ast_eval.py . Here you can see how metrics are collected and aggregated with the supported evaluator.

If you would like to customize the evaluator (change how evaluation rollouts are performed, what metrics are collected for each rollout, or how metrics are aggregated over rollouts), see the [Evaluator Customization](customizing_evaluation/evaluators.md) guide.