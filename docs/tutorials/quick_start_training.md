# Quick Start: Training

Do you want to train a HuggingFace **auditor** using an ASTRA-supported algorithm (e.g., DPO, IPO, PPO) and system formulation ([ASTPrompter](https://arxiv.org/abs/2407.09447), [Perez et al.](https://aclanthology.org/2022.emnlp-main.225/), MALIBU*, [CRT*](https://arxiv.org/abs/2402.19464))? *coming soon


Then this guide is for you. We'll walk through every step required to train an adversarial testing auditor (llama3) using our pre-configured classes and point you to customization guides when your use case goes beyond the defaults. By using our pre-configured classes, you'll be training your auditor in 7 easy steps!

---

## Step 1: Setup

Please see the [main documentation](../index.md) for full setup instructions. Here's a quick recap:

```bash
# Install the ASTRA-RL toolbox
pip install astra-rl

# Import ASTRA-RL in your python code
import astra_rl
```

!!! note
    wandb is not automatically installed during this process. If you would like to use wandb, either install it as an optional dependency (`pip install "astra-rl[wandb]"`) or install it directly (`uv pip install wandb`) and run export `WANDB_API_KEY=YOUR_API_KEY_HERE` in your terminal.

---

## Step 2: Import Required Modules and set device

```python
from torch.optim import AdamW

# ASTRA-RL core components
from astra_rl import ASTSampler, DPO, DetoxifyScorer, Harness

# HuggingFace-friendly system wrapper for ASTPrompter-style red teaming
from astra_rl.ext.transformers import HFASTSystem
from astra_rl.training import Trainer, TrainingConfiguration

DEVICE = "cuda"  # or "cpu" if GPU is not available
```
We support both lightweight (e.g., GPT-2 + Detoxify) and heavyweight (e.g., LLaMA + LlamaGuard) setups. Pick model and scorer sizes that fit your compute!

---

## Step 3: Load Your Initial Prompts
To train an auditor, you'll need a list of comma-separated strings that act as initial prompts—these initiate auditor-target rollouts used for online training. 

```python
import json

with open("prompts_reddit_train.json") as f:
    PROMPTS = json.load(f)
```

Since ASTPrompter tests for harmful outputs in conversational settings, it uses the ConvoKit Reddit Small Corpus (filtered for proper formatting and for non-toxicity using Detoxify) as its default source of initial prompts. This data can be found in the GPT2_v_GPT2 folder in examples.

The ASTRA-RL toolbox easily supports external prompt datasets or APIs—just ensure the final PROMPTS variable is formatted as a list of strings.

---

## Step 4: Instantiate Your System

The *system* is an important component of training that handles rollout step generation, reward computation, and log-probability calculation. To speed you along, we have implemented the `HFASTSystem` class that handles the technical backend so you just need to provide the huggingface model IDs of the auditor, target and baseline models and a `Scorer` instance (DetexifyScorer(), LlamaGuardScorer() or your custom scorer).

> Note: Your auditor and target can be different models but your baseline and auditor should typically be the same.

> Note: HFASTSystem can not handle huggingface models that have a fixed length (e.g. GPT2) as it will run into an out of context error. If you would like to train a GPT2-based auditor model, please see our GPT2_v_GPT2 examples which use a custom ExampleDetoxifySystem() designed for GPT2 models.

```python
# Example system instantiation: llama3 auditor, target, and baseline with Detoxify scorer (heavyweight setup - requires GPU)
system = HFASTSystem("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", DetoxifyScorer(), DEVICE)
```

Need a custom model or rollout step logic? See the [System Customization](customizing_training/problems.md) guide. Want to use a custom scorer? See the [Scorer Customization](customizing_training/moderators.md) guide.

---

## Step 5: Instantiate the Sampler

The sampler defines how training rollouts are structured and collected. In ASTRA-RL, the default is the `ASTSampler`, which implements the conversation tree rollout used in the [ASTPrompter](https://arxiv.org/abs/2407.09447) paper.

```python
sampler = ASTSampler(system, PROMPTS)
```

<details>
  <summary><strong>Curious about how this sampler structures rollouts?</strong></summary>

  This sampler builds a tree-structured conversation graph, where:
  - The root node starts from a random initial prompt (from `PROMPTS`)
  - At each turn, the auditor generates multiple (`tree_width`, default 2) candidate utterances
  - Each of those utterances is fed to the target model, which produces a response
  - The resulting auditor–target tuples form child nodes
  - This process repeats for `tree_depth` levels (default 3), yielding a multi-turn auditor–target dialogue tree.

  This structure enables preference-based learning algorithms like DPO and IPO to reason over multiple conversational branches at once, training the auditor to elicit harmful responses in a multi-turn setting.
</details>


By default, rollouts are configured with tree_width=2 and tree_depth=3, but you can customize both:
```python
sampler = ASTSampler(system, PROMPTS, tree_width=4, tree_depth=5)
```

Want a different rollout graph structure or a multi-agent setup? See the [Sampler Customization](customizing_training/environments.md) guide.

---

## Step 6: Choose Your Algorithm and Optimizer

The solver is the RL learning algorithm that will take in a graph of training rollouts and compute the loss. The optimizer will update auditor model weights
to minimize this loss, teaching the auditor to more effectively ellicit target toxicity.

We use DPO and Adam as the default for this quickstart.

```python
solver = DPO(system)
optimizer = AdamW(system.parameters(), lr=1e-5)
```

To integrate your own RL algorithm, see the [Solver Customization](customizing_training/solvers.md) guide.

---

## Step 7: Train the Auditor

For the quick start approach, simply call our training configuration and trainer classes and start training!

```python
# instantiate the pre-configured HF-compatable configuration and traininer class
config = HFASTConfiguration() # lr = 1e-5, batch size = 4, optimizer = "adamw", no gradient accumulation, 1000 training steps, 2 episodes per experience
# this trainer will train the auditor and evaluate it on a dev set every 100 steps, saving the best model to "checkpoints"
trainer = HFASTTrainer(
    config,
    sampler,
    solver,
    dev_prompts=DEV_PROMPTS,
    eval_every=100,
    ckpt_dir="checkpoints",
)
trainer.train()
```
> The source code for the training configuration and trainer are at [hf_ast_system](https://github.com/sisl/astra-rl/blob/main/src/astra_rl/ext/transformers/hf_ast_system.py)

Want to customize the training configuration/hyperparams, the training loop, or model saving/eval? See the [Trainer Customization](customizing_training/trainers.md) guide.

---

## Full Examples:
We provide 3 complete working examples that mirror this guide!

Hugging face example for llama3 models without trainer: [examples/ast_huggingface.py](https://github.com/sisl/astra-rl/blob/main/examples/ast_huggingface.py)

Custom AST system for GPT2 models with trainer: [examples/ast_trainer](https://github.com/sisl/astra-rl/blob/main/examples/GPT2_v_GPT2/ast_trainer.py)

Custom AST system for GPT2 models without trainer: [examples/ast_basic.py](https://github.com/sisl/astra-rl/blob/main/examples/ast_basic.py)

