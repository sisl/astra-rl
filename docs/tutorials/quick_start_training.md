# Quick Start: Training a HuggingFace Attacker with ASTRA-RL

Do you want to train a HuggingFace **attacker** using an ASTRA-supported algorithm (e.g., DPO, IPO, PPO) and problem formulation ([ASTPrompter](https://arxiv.org/abs/2407.09447), [RL - Perez*](https://aclanthology.org/2022.emnlp-main.225/), MALIBU*, [CRT*](https://arxiv.org/abs/2402.19464))?

Then this guide is for you. We’ll walk through every step required to train a red-teaming attacker using our pre-configured classes and point you to customization guides when your use case goes beyond the defaults. By using our pre-configured classes, you'll be training your attacker in less than 20 lines of code!

---

## Step 1: Setup

Please see the [README](../../README.md) for full setup instructions. Here's a quick recap:

```bash
# Install the ASTRA-RL toolbox
pip install astra-rl

# Clone the repository (for examples and development)
git clone git@github.com:sisl/astra-rl.git
```

>Note: wandb is not automatically installed during this process. If you would let to use wandb (supported), install it (uv pip install wandb) and run export WANDB_API_KEY=###your_wandb_api_key##### in your terminal.

---

## Step 2: Import Required Modules and set device

```python
from torch.optim import AdamW

# ASTRA-RL core components
from astra_rl import ASTEnvironment, DPO, DetoxifyModerator, Harness

# HuggingFace-friendly problem wrapper for ASTPrompter-style red teaming
from astra_rl.ext.transformers import HFASTProblem

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

Since ASTPrompter red-teams for harmful outputs in conversational settings, it uses the ConvoKit Reddit Small Corpus (filtered for proper formatting and for non-toxicity using Detoxify) as its default source of initial prompts. This data can be found in astra-rl/examples/GPT2_v_GPT2/.

The ASTRA-RL toolbox easily supports external prompt datasets or APIs—just ensure the final PROMPTS variable is formatted as a list of strings.

---

## Step 4: Instantiate Your Problem

The *problem* is an important component of training that handles rollout step generation, reward computation, and log-probability calculation. To speed you along, we have implemented the `HFASTProblem` class that handles the technical backend so you just need to provide the huggingface model IDs of the attacker, target and baseline models and a `Moderator` instance (DetexifyModerator(), LlamaGuardModerator() or your own custom one).

Note: Your attacker and target can be different models (e.g., GPT-2 attacker, LLaMA target) but your baseline and attacker should typically be the same.

```python
from astra_rl.modifiers import LlamaGuardModerator  # optional

# Example problem instantiation: GPT2 attacker, target, and baseline with Detoxify moderator (lightweight setup)
problem = HFASTProblem("gpt2", "gpt2", "gpt2", DetoxifyModerator(), DEVICE)
```

Need a custom model or rollout step logic? See [customize_training/problems](astra-rl/docs/tutorials/customize_training/problems.md)

Want to use a custom moderator? See [customize_training/moderators](astra-rl/docs/tutorials/customize_training/moderators.md)

---

## Step 5: Instantiate the Environment

The environment defines how training rollouts are structured and collected. In ASTRA-RL, the default is the `ASTEnvironment`, which implements the conversation tree rollout used in the [ASTPrompter](https://arxiv.org/abs/2407.09447) paper.

```python
env = ASTEnvironment(problem, PROMPTS)
```

<!-- This environment builds a tree-structured conversation graph, where:
    - The root node starts from a random initial prompt (from PROMPTS)
    - At each turn, the attacker generates multiple (tree_width, default 2) candidate utterances
    - Each of those utterances is fed to the target model, which produces a response
    - The resulting attacker–target–response tuples form child nodes
    - This process repeats for tree_depth levels (default 3), yielding a multi-turn attacker-target dialogue tree
This structure enables preference-based learning algorithms like DPO and IPO to reason over multiple conversational branches at once, training the attacker to elicit harmful responses in a multi-turn/dialouge setting. -->
<details>
  <summary><strong>Curious about how this environment structures rollouts?</strong></summary>

  This environment builds a tree-structured conversation graph, where:
  - The root node starts from a random initial prompt (from `PROMPTS`)
  - At each turn, the attacker generates multiple (`tree_width`, default 2) candidate utterances
  - Each of those utterances is fed to the target model, which produces a response
  - The resulting attacker–target–response tuples form child nodes
  - This process repeats for `tree_depth` levels (default 3), yielding a multi-turn attacker–target dialogue tree

  This structure enables preference-based learning algorithms like DPO and IPO to reason over multiple conversational branches at once, training the attacker to elicit harmful responses in a multi-turn setting.
</details>


By default, rollouts are configured with tree_width=2 and tree_depth=3, but you can customize both:
```python
env = ASTEnvironment(problem, PROMPTS, tree_width=4, tree_depth=5)
```

Want a different rollout graph structure or a multi-agent setup? See [customize_training/environments](/home/allie11/astra-rl/docs/tutorials/customize_training/envirnoments.md)

---

## Step 6: Choose Your Algorithm and Optimizer

The solver is the RL learning algorithm that will take in a graph of training rollouts and compute the training loss. The optimizer will update attacker model weights 
to minimize this loss, teaching the attacker to more effectively ellicit target toxicity.

We use DPO and Adam as the default for this quickstart.

```python
solver = DPO(problem)
optimizer = AdamW(problem.parameters(), lr=1e-5)
```

 To integrate your own RL algorithm, see [customize_training/solvers](/home/allie11/astra-rl/docs/tutorials/customize_training/solvers.md)

---

## Step 7: Create the Training Harness

The training harness wires your environment and solver together. It collects online experience and, for each batch, invokes the solver to compute the loss used to update the attacker’s policy. You typically won’t need to modify the harness code itself—adjust behavior via the environment, solver, or your outer training loop (e.g., schedules, logging, hyperparameters).

```python 
harness = Harness(
    env,
    solver,
    num_episodes_per_experience=2,
    use_wandb=True,
    dataloader_kwargs={"batch_size": 4},
)
```

## Step 8: Train the Attacker

Choose how many training steps/epochs to run. You have full control over the loop—curriculum, evaluation cadence, checkpointing, learning-rate schedules, gradient clipping, and more.

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

---

## Full Example: examples/ast_hf.py
We provide a complete working example that mirrors this guide!
