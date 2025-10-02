# Tutorials

Welcome to ASTRA-RL! This section provides step-by-step guides and examples to help you get started with the ASTRA-RL toolbox.

ASTRA-RL is a user-friendly, modular, and customizable toolbox for **language model testing and evaluation**. It's designed for quickly getting started evaluating models or building your own adversarial testing pipelines for training and evaluation.

---

## What is Language Model Adversarial Testing?

**Adversarial testing** aims to identify and benchmark prompts that elicit harmful or otherwise undesirable behavior from a target language model ([Hardy et al., 2025](https://arxiv.org/abs/2407.09447)). This surfaces vulnerabilities and guides fine-tuning to reduce harmful outputs.

* **Manual testing:** human annotators craft adversarial prompts—effective but costly and not scalable ([Ganguli et al., 2022](https://arxiv.org/abs/2209.07858)).
* **Automated testing:** generates adversarial prompts at scale. Examples include fuzzing existing prompts ([Yu et al., 2023](https://arxiv.org/abs/2309.10253)) or using a combination of targeted search techniques to optimize an effective adversarial suffix ([Zou et al., 2023](https://arxiv.org/abs/2307.15043))

### RL-based Adversarial Testing

A promising direction in automated testing is **reinforcement learning (RL)**. We train a separate **attacker** policy (often an LLM) to maximize a **non-differentiable reward** (e.g., toxicity) computed on the target's response. In short, we **automate** adversarial testing by training an attacker to generate test cases that increase the chance of unsafe target outputs.

**Pros**

1. **Fast at inference:** once trained, generating new prompts is quick and inexpensive.
2. **Effective:** prior work (e.g., Perez; Huang; Hardy) shows RL-trained attackers reliably induce harmful behavior.

**Cons**

1. **Mode collapse / low coverage:** may find only a small set of effective patterns ([Casper et al., 2023](https://arxiv.org/abs/2306.09442)).
2. **Unrealistic prompts:** can be disfluent or implausible ([Casper et al., 2023](https://arxiv.org/abs/2306.09442); [Deng et al., 2022](https://arxiv.org/abs/2205.12548)), even with realism terms ([Wichers et al., 2024](https://arxiv.org/abs/2401.16656)).

ASTRA-RL makes **training** an RL-based attacker and **evaluating** a target with a pre-trained attacker both **quick** and **customizable**.

---

## Key Terminology

* **Target** (a.k.a. *defender*, *model under test*)
  The model being evaluated. It converses with the attacker; the target's response is scored by a **moderator**, and that score contributes to the attacker's reward.

* **Attacker**
  The policy (often an LLM) that generates utterances intended to elicit harmful responses from the target. Typically initialized from a general LM (e.g., Llama-2) and **updated via RL** to improve effectiveness.

* **Moderator**
  The scoring component (like a reward model). At each step, it returns a **scalar measure of harm** (e.g., toxicity). "Harm" can be defined via existing classifiers (e.g., Llama-Guard 3) or a custom model you provide.

---

## Package Overview

ASTRA-RL decomposes RL-based adversarial testing into five pieces. 

### 1) Problem — *How models run/interact*

Handles loading models/tokenizers, performing **one rollout step** (attacker/target generation), computing **log-probs** (for DPO/PPO, etc.), advancing the conversation state, and defining a **reward**. See the [Problem Customization](customizing_training/problems.md) guide to learn more.

### 2) Environment — *How data is collected*

Defines how attacker–target interactions are generated and structured for training/eval (e.g., **single-path vs. tree** rollouts), what per-step data is stored, and what the solver receives. See the [Environment Customization](customizing_training/environments.md) guide to learn more.

### 3) Moderators — *How we define/measure harm*

Scores target generations (scalar harm). Instantiate in your **Problem** for seamless use. See the [Moderator Customization](customizing_training/moderators.md) guide to learn more.

### 4) Solvers (Algorithms) — *How the attacker learns*

Consume rollout graphs, **flatten** them to per-sample steps, **collate** batches, and compute the **training loss** (plus logs). See the [Solver Customization](customizing_training/solvers.md) guide to learn more.

### 5) Trainer — *How the training loop runs*

Orchestrates the main loop, hyperparameters, optimizer, eval cadence, and checkpointing. See the [Trainer Customization](customizing_training/trainers.md) guide to learn more.

---

## Quick Start

### Train an RL-Based Attacker

Start here to train with supported moderators, algorithms, and HF attackers/defenders:
**[Quick Start Training](quick_start_training.md)**

**Supported out-of-the-box**

| Component                        | Options                                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Training Algorithms              | PPO, DPO, IPO                                                                                                          |
| Moderators                       | [Llama-Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B), [Detoxify](https://github.com/unitaryai/detoxify) |
| Problem Formulations (published) | [ASTPrompter](https://arxiv.org/abs/2407.09447), [Perez et al.](https://aclanthology.org/2022.emnlp-main.225/)           |

Want to go beyond the defaults? ASTRA-RL is **modular**—swap what you need and reuse everything else. We recommend starting with the quick start to learn the overall flow; it links directly to the relevant **customization guides** when you diverge.

### Evaluate a Target with a Pre-Trained Attacker

Jump straight to evaluation (single-path dev/test rollouts, metric aggregation):
**[Quick Start Evaluation](quick_start_evaluation.md)**

