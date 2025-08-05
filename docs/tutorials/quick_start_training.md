# Quick Start: Training a HuggingFace Attacker with ASTRA-RL

Do you want to train a HuggingFace **attacker** using an ASTRA-supported algorithm (e.g., DPO, IPO, PPO*) and problem formulation (ASTPrompter, Perez*, MALIBU*, Hong*)?

Then this guide is for you. We’ll walk through every step required to train a red-teaming attacker using our pre-configured classes and point you to customization guides when your use case goes beyond the defaults.

---

## Step 1: Setup

Please see the [README](../../README.md) for full setup instructions. Here's a quick recap:

```bash
# Install the ASTRA-RL toolbox
pip install astra-rl

# Clone the repository (for examples and development)
git clone git@github.com:sisl/astra-rl.git
cd astra-rl

# Sync dependencies
uv sync --dev

# (Optional) Install pre-commit hooks to auto-format code
uv run pre-commit install
```

---

## Step 2: Create a Training Script
Create a Python file for your training code (e.g., train.py).

---

## Step 3: Import Required Modules

```python
from torch.optim import AdamW

# ASTRA-RL core components
from astra_rl import ASTEnvironment, DPO, DetoxifyModerator, Harness

# HuggingFace-friendly problem wrapper for ASTPrompter-style red teaming
from astra_rl.ext.transformers import HFASTProblem
```

## Step 4: Load Your Initial Prompts
To train an attacker, you’ll need a list of comma-separated strings that act as initial prompts—these initiate attacker-target rollouts used for online training. 

```python
import json

with open("prompts_reddit_train.json") as f:
    PROMPTS = json.load(f)
```

Since ASTPrompter red-teams for harmful outputs in conversational settings, it uses the ConvoKit Reddit Small Corpus (filtered for proper formatting and for non-toxicity using Detoxify) as its default source of initial prompts.

The ASTRA-RL toolbox also supports external prompt datasets or APIs—just ensure the final PROMPTS variable is formatted as a list of strings.

---

## Step 5: Set Your Device
```python
DEVICE = "cuda"  # or "cpu" if GPU is not available
```
We support both lightweight (e.g., GPT-2 + Detoxify) and heavyweight (e.g., LLaMA + LlamaGuard) setups.

---

## Step 6: Instantiate Your Problem

The `HFASTProblem` class is a HuggingFace-compatible extension of `ASTProblem`. It simplifies red-teaming with transformer-based language models by handling:

- Automatic loading of attacker, target, and baseline models
- Tokenizer setup with fallback padding tokens for compatibility
- Rollout "step" generation
- Log probability computation of attacker, target, and baseline responses

You simply provide the model IDs and a `Moderator` instance, and `HFASTProblem` manages the rest—making it easy to plug into ASTRA-RL’s training and evaluation pipeline.

Note: If your attacker and target are different models (e.g., GPT-2 attacker, LLaMA target), this class handles all the tokenizer/model interop for you.


```python
from astra_rl.modifiers import LlamaGuardModerator  # optional

# Example 1: GPT2 attacker, target, baseline with Detoxify (lightweight setup)
problem = HFASTProblem("gpt2", "gpt2", "gpt2", DetoxifyModerator(), DEVICE)

# Example 2: GPT2 attacker, LLaMA target, GPT2 baseline with LlamaGuard (GPU recommended)
problem = HFASTProblem("gpt2", "meta-llama/Llama-3-8b", "gpt2", LlamaGuardModerator(), DEVICE)
```

Need a custom model or rollout step logic? See
    -> astra-rl/docs/tutorials/customize/problems

Want to use a custom moderator? See
    -> astra-rl/docs/tutorials/customize/moderators

---

## Step 7: Instantiate the Environment

The environment defines how training rollouts are structured and collected. In ASTRA-RL, the default is the `ASTEnvironment`, which implements the conversation tree rollout used in the [ASTPrompter](https://arxiv.org/abs/2407.09447) paper.

```python
env = ASTEnvironment(problem, PROMPTS)
```

This environment builds a tree-structured conversation graph, where:
    - The root node starts from a random initial prompt (from PROMPTS)
    - At each turn, the attacker generates multiple candidate utterances (tree_width, default 2)
    - Each of those utterances is fed to the target model, which produces a response
    - The resulting attacker–target–response tuples form child nodes
    - This process repeats for tree_depth levels (default 3), yielding a multi-turn attacker-target dialogue tree
This structure enables preference-based learning algorithms like DPO and IPO to reason over multiple conversational branches at once, training the attacker to elicit increasingly harmful responses over time.

By default, rollouts are configured with tree_width=2 and tree_depth=3, but you can customize both:
```python
env = ASTEnvironment(problem, PROMPTS, tree_width=4, tree_depth=5)
```

Want a different rollout structure or multi-agent setup? See
    -> astra-rl/docs/tutorials/customize/environments

---

## Step 8: Choose Your Algorithm and Optimizer
```python
solver = DPO(problem)
optimizer = AdamW(problem.parameters(), lr=1e-5)
```

 To integrate your own RL algorithm, see
    -> astra-rl/docs/tutorials/customize/solvers

---

## Step 9: Create the Training Harness

```python 
harness = Harness(
    env,
    solver,
    num_episodes_per_experience=2,
    use_wandb=True,
    dataloader_kwargs={"batch_size": 4},
)
```
For more information on the training harness, see
    -> astra-rl/docs/tutorials/customize_training/harness

---

## Step 10: Train the Attacker

```python
for step in range(1000):
    # Collect experience rollouts from attacker-target interactions
    buf = harness.experience()
    for experience in buf:
        # Compute loss using the solver (e.g., DPO)
        loss, step_logs = harness.step(experience)

        # Standard PyTorch optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step_logs["step"] = step
        harness.log_current_step(step_logs)
```

## Full Example: examples/ast_hf.py
We provide a complete working example that mirrors this guide!
