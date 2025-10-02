# Quick Start: Evaluation

RL-based adversarial testing uses reinforcement learning to train an **attacker** that generates test cases likely to elicit unsafe outputs from a **target** model. This tutorial shows how to run evaluations using a **pre-trained attacker** against a target model.

!!! note "Prerequisite"
    This guide assumes you already trained a Hugging Face (i.e. llama3) attacker (see **Quick Start: Training**). You’ll point evaluation at that saved attacker checkpoint.

---

## Quick Start

Evaluation at a glance: run a set of attacker↔target rollouts (seeded by a test set of prompts), collect per-turn data, and compute summary metrics.

### 1) Setup: imports, model paths, and device

Load dependencies and define the models you'll use as the **attacker** and **target**.

```python
# import dependencies 
import torch
import json
from astra_rl import DetoxifyModerator, ASTEnvironment
from astra_rl.methods.ast_problem import ASTEvaluator
from astra_rl.ext.transformers.hf_ast_problem import HFEvaluationProblem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your attacker model checkpoint from training
ATTACKER_MODEL = "/home/user/astra-rl/examples/checkpoints/best" # assuming tokenizer is in checkpoint (default save in training)
TARGET_MODEL = "meta-llama/Llama-3.1-8B"  # can be any HF model
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

Use the preconfigured **HFEvaluationProblem** (takes in a pre-trained HF model) and the **ASTEnvironment** (ASTPrompter-style rollouts).

```python
# instantiate the HF evaluation problem with your attacker and target models
problem = HFEvaluationProblem(
    attacker_checkpoint=ATTACKER_MODEL,      # local HF dir or Hub id with trained attacker
    attacker_base_model_id=None,             # optional fallback; None if checkpoint includes tokenizer
    target_model_id=TARGET_MODEL,            # HF id of the target model
    device=DEVICE,                           # "cuda" or "cpu"
    moderator=DetoxifyModerator(),           # harm classifier (optional)
)

# instantiate the AST environment — no adjustments needed because eval_rollout is available
env = ASTEnvironment(problem, PROMPTS, tree_width=1, tree_depth=3)
```


??? info "HFEvaluation inputs"
    - `ATTACKER_MODEL` — Path or Hugging Face hub id for your trained attacker checkpoint (e.g. `"/home/user/checkpoints/best"` or `"username/attacker-checkpoint"`). The checkpoint may include model weights **and** tokenizer files.
    - `ATTACKER_BASE_MODEL` *(optional)* — Hugging Face model id used as the **base tokenizer** at training-time (e.g. `"meta-llama/Llama-3.1-8B"`). Only required if the checkpoint **does not** include tokenizer files.
    - `TARGET_MODEL` — Hugging Face id of the model you want to evaluate (e.g. `"meta-llama/Llama-3.1-8B"`).
    - `DEVICE` — Device string where you want to load models/tokenizers (e.g. `"cuda"` or `"cpu"`).
    - `MODERATOR` — Moderator instance that defines the harm metric (e.g. `DetoxifyModerator()` or `LlamaGuardModerator()`).

!!! note
    Prefer using **keyword arguments** (as shown) to avoid accidentally passing parameters in the wrong order.  

    If your attacker checkpoint includes tokenizer files (common when saving via `model.save_pretrained()` + `tokenizer.save_pretrained()` during training), set `attacker_base_model_id=None` — the evaluation class will load the tokenizer from the checkpoint. If the checkpoint lacks tokenizer files, provide `attacker_base_model_id` so the correct tokenizer can be loaded.

!!! tip
    If you want to evaluate GPT-2 attackers, use `GPT2EvaluationProblem` instead of `HFEvaluationProblem`.  
    See the full GPT-2 evaluation example: [`gpt2_eval.py`](https://github.com/sisl/astra-rl/blob/main/examples/gpt2_eval.py).


If you want to evaluate a custom attacker or target, create a new `Problem` subclass. See the [Evaluation Problem Customization](customizing_evaluation/evaluation_problems.md) guide.

### 4) Evaluate!

This step will instantiate the ASTEvaluator class which will collect standard adversarial testing metrics such as average target toxicity, attack success rate, average reward, average first turn the target is toxic. 

Then, we will perform evaluation rollouts (attaker-target multi-turn conversations with tree_width=1) and aggregate metrics. The metrics will be saved to your repository as a json. 

```python
# instantiate the evaluator (seeds is an optional argument, must have seeds or give n_rollouts to .evaluate below)
evaluator = ASTEvaluator(env, seeds=PROMPTS)

# collect metrics by running n_rollouts
metrics = evaluator.evaluate(n_rollouts=200, progress=True)

# save metrics to json file
evaluator.write_json(metrics, "metrics.json")
```

??? info "What metrics does `ASTEvaluator` collect?"
    **Per-turn metrics** *(one entry per conversation step)*

    - `turn` — turn number in the rollout  
    - `reward` — environment reward at that turn  
    - `adversarial_toxicity` — toxicity of the attacker’s message  
    - `target_toxicity` — toxicity of the target’s response  
    - `context` — truncated conversation context  
    - `attack` — attacker’s utterance  
    - `response` — target’s response  

    **Overall rollout metrics**

    - `num_turns` — number of turns in the rollout  
    - `avg_reward` — mean reward across turns  
    - `avg_adversarial_toxicity` — average toxicity of attacker turns  
    - `avg_target_toxicity` — average toxicity of target turns  
    - `first_adversarial_toxicity_turn` — earliest turn attacker became toxic  
    - `first_target_toxicity_turn` — earliest turn target became toxic  

    **Aggregated metrics** *(across all rollouts)*

    - `avg_reward` — average reward per rollout  
    - `avg_adversarial_toxicity` — average attacker toxicity across rollouts  
    - `avg_target_toxicity` — average target toxicity across rollouts  
    - `pct_rollouts_with_adversarial_toxicity` — fraction of rollouts where attacker was toxic at least once  
    - `pct_rollouts_with_target_toxicity / attack_success_rate` — fraction of rollouts where target became toxic  

!!! note
    The source code for ASTEvaluator is located at [methods/ast_problem](https://github.com/sisl/astra-rl/blob/main/src/astra_rl/methods/ast_problem.py). Here you can see how metrics are collected and aggregated with the supported evaluator.

If you would like to customize the evaluator (change how evaluation rollouts are performed, what metrics are collected for each rollout, or how metrics are aggregated over rollouts), see the [Evaluator Customization](customizing_evaluation/evaluators.md) guide.