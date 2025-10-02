# Quick Start: Training a HuggingFace Attacker with ASTRA-RL

Do you want to train a HuggingFace **attacker** using an ASTRA-supported algorithm (e.g., DPO, IPO, PPO) and problem formulation ([ASTPrompter](https://arxiv.org/abs/2407.09447), [Perez et al.](https://aclanthology.org/2022.emnlp-main.225/), MALIBU*, [CRT*](https://arxiv.org/abs/2402.19464))? *coming soon


Then this guide is for you. We'll walk through every step required to train an adversarial testing attacker (llama3) using our pre-configured classes and point you to customization guides when your use case goes beyond the defaults. By using our pre-configured classes, you'll be training your attacker in 7 easy steps!

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
from astra_rl import ASTEnvironment, DPO, DetoxifyModerator, Harness

# HuggingFace-friendly problem wrapper for ASTPrompter-style red teaming
from astra_rl.ext.transformers import HFASTProblem
from astra_rl.training import Trainer, TrainingConfiguration

DEVICE = "cuda"  # or "cpu" if GPU is not available
```
We support both lightweight (e.g., GPT-2 + Detoxify) and heavyweight (e.g., LLaMA + LlamaGuard) setups. Pick model and moderator sizes that fit your compute!

---

## Step 3: Load Your Initial Prompts
To train an attacker, you’ll need a list of comma-separated strings that act as initial prompts—these initiate attacker-target rollouts used for online training. 

```python
import json

with open("prompts_reddit_train.json") as f:
    PROMPTS = json.load(f)
```

Since ASTPrompter tests for harmful outputs in conversational settings, it uses the ConvoKit Reddit Small Corpus (filtered for proper formatting and for non-toxicity using Detoxify) as its default source of initial prompts. This data can be found in the GPT2_v_GPT2 folder in examples.

The ASTRA-RL toolbox easily supports external prompt datasets or APIs—just ensure the final PROMPTS variable is formatted as a list of strings.

---

## Step 4: Instantiate Your Problem

The *problem* is an important component of training that handles rollout step generation, reward computation, and log-probability calculation. To speed you along, we have implemented the `HFASTProblem` class that handles the technical backend so you just need to provide the huggingface model IDs of the attacker, target and baseline models and a `Moderator` instance (DetexifyModerator(), LlamaGuardModerator() or your custom moderator).

> Note: Your attacker and target can be different models but your baseline and attacker should typically be the same.

> Note: HFASTProblem can not handle huggingface models that have a fixed length (e.g. GPT2) as it will run into an out of context error. If you would like to train a GPT2-based attacker model, please see our GPT2_v_GPT2 examples which use a custom ExampleDetoxifyProblem() designed for GPT2 models.

```python
# Example problem instantiation: llama3 attacker, target, and baseline with Detoxify moderator (heavyweight setup - requires GPU)
problem = HFASTProblem("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", DetoxifyModerator(), DEVICE)
```

Need a custom model or rollout step logic? See the [Problem Customization](customizing_training/problems.md) guide. Want to use a custom moderator? See the [Moderator Customization](customizing_training/moderators.md) guide.

---

## Step 5: Instantiate the Environment

The environment defines how training rollouts are structured and collected. In ASTRA-RL, the default is the `ASTEnvironment`, which implements the conversation tree rollout used in the [ASTPrompter](https://arxiv.org/abs/2407.09447) paper.

```python
env = ASTEnvironment(problem, PROMPTS)
```

<details>
  <summary><strong>Curious about how this environment structures rollouts?</strong></summary>

  This environment builds a tree-structured conversation graph, where:
  - The root node starts from a random initial prompt (from `PROMPTS`)
  - At each turn, the attacker generates multiple (`tree_width`, default 2) candidate utterances
  - Each of those utterances is fed to the target model, which produces a response
  - The resulting attacker–target tuples form child nodes
  - This process repeats for `tree_depth` levels (default 3), yielding a multi-turn attacker–target dialogue tree.

  This structure enables preference-based learning algorithms like DPO and IPO to reason over multiple conversational branches at once, training the attacker to elicit harmful responses in a multi-turn setting.
</details>


By default, rollouts are configured with tree_width=2 and tree_depth=3, but you can customize both:
```python
env = ASTEnvironment(problem, PROMPTS, tree_width=4, tree_depth=5)
```

Want a different rollout graph structure or a multi-agent setup? See the [Environment Customization](customizing_training/environments.md) guide.

---

## Step 6: Choose Your Algorithm and Optimizer

The solver is the RL learning algorithm that will take in a graph of training rollouts and compute the loss. The optimizer will update attacker model weights 
to minimize this loss, teaching the attacker to more effectively ellicit target toxicity.

We use DPO and Adam as the default for this quickstart.

```python
solver = DPO(problem)
optimizer = AdamW(problem.parameters(), lr=1e-5)
```

To integrate your own RL algorithm, see the [Solver Customization](customizing_training/solvers.md) guide.

---

## Step 7: Train the Attacker

For the quick start approach, simply call our training configuration and trainer classes and start training!

```python
# instantiate the pre-configured HF-compatable configuration and traininer class
config = HFASTConfiguration() # lr = 1e-5, batch size = 4, optimizer = "adamw", no gradient accumulation, 1000 training steps, 2 episodes per experience
# this trainer will train the attacker and evaluate it on a dev set every 100 steps, saving the best model to "checkpoints"
trainer = HFASTTrainer(
    config,
    env,
    solver,
    dev_prompts=DEV_PROMPTS,
    eval_every=100,
    ckpt_dir="checkpoints",
)
trainer.train()
```
> The source code for the training configuration and trainer are at [hf_ast_problem](https://github.com/sisl/astra-rl/blob/main/src/astra_rl/ext/transformers/hf_ast_problem.py)

Want to customize the training configuration/hyperparams, the training loop, or model saving/eval? See the [Trainer Customization](customizing_training/trainers.md) guide.

---

## Full Examples: 
We provide 3 complete working examples that mirror this guide!

Hugging face example for llama3 models without trainer: [examples/ast_hf.py](https://github.com/sisl/astra-rl/blob/main/examples/ast_hf.py)

Custom AST problem for GPT2 models with trainer: [examples/ast_trainer](https://github.com/sisl/astra-rl/blob/main/examples/GPT2_v_GPT2/ast_trainer.py)

Custom AST problem for GPT2 models without trainer: [examples/ast_basic.py](https://github.com/sisl/astra-rl/blob/main/examples/ast_basic.py)

