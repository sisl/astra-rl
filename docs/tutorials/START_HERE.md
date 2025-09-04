# Welcome to ASTRA-RL!

ASTRA-RL is a user-friendly, modular, and customizable toolbox for **LM red-teaming**—great for quickly getting started or swapping in your own research components.

---

## Table of Contents

1. [What is LM Red-Teaming?](#1-what-is-lm-red-teaming)

   * [1.1 RL-based Red-Teaming](#11-rl-based-red-teaming)
2. [Key Terminology](#2-key-terminology)
3. [Package Overview](#3-package-overview)
4. [Quick Start](#4-quick-start)

   * [4.1 Train an RL-Based Attacker](#41-train-an-rl-based-attacker)
   * [4.2 Evaluate a Target with a Pre-Trained Attacker](#42-evaluate-a-target-with-a-pre-trained-attacker)
5. [References](#5-references)
6. [Concept Notes](#6-concept-notes)

---

## 1. What is LM Red-Teaming?

**LM red-teaming** aims to identify and benchmark prompts that elicit harmful or otherwise undesirable behavior from a target language model (Hardy et al., 2025). This surfaces vulnerabilities and guides fine-tuning to reduce harmful outputs.

* **Manual red-teaming:** human annotators craft adversarial prompts—effective but costly and not scalable (Ganguli et al., 2022).
* **Automated red-teaming:** generates adversarial prompts at scale. Examples include fuzzing exisiting prompts (Yu et al., 2023) or using a combination of targeted search techniques to optimize an effective adversarial suffix (Zou et al., 2023)

### 1.1 RL-based Red-Teaming

A promising direction in automated red-teaming is **reinforcement learning (RL)**. We train a separate **attacker** policy (often an LLM) to maximize a **non-differentiable reward** (e.g., toxicity) computed on the target’s response. In short, we **automate** red-teaming by training an attacker to generate test cases that increase the chance of unsafe target outputs.

**Pros**

1. **Fast at inference:** once trained, generating new prompts is quick and inexpensive.
2. **Effective:** prior work (e.g., Perez; Huang; Hardy) shows RL-trained attackers reliably induce harmful behavior.

**Cons**

1. **Mode collapse / low coverage:** may find only a small set of effective patterns (Casper et al., 2023).
2. **Unrealistic prompts:** can be disfluent or implausible (Deng et al., 2022; Casper et al., 2023), even with realism terms (Wichers et al., 2024).

ASTRA-RL makes **training** an RL attacker and **evaluating** a target with a pre-trained attacker both **quick** and **customizable**.

---

## 2. Key Terminology

* **Target** (a.k.a. *defender*, *model under test*)
  The model being red-teamed. It converses with the attacker; the target’s response is scored by a **moderator**, and that score contributes to the attacker’s reward.

* **Attacker**
  The policy (often an LLM) that generates utterances intended to elicit harmful responses from the target. Typically initialized from a general LM (e.g., Llama-2) and **updated via RL** to improve effectiveness.

* **Moderator**
  The scoring component (like a reward model). At each step, it returns a **scalar measure of harm** (e.g., toxicity). “Harm” can be defined via existing classifiers (e.g., Llama-Guard 3) or a custom model you provide.

---

## 3. Package Overview

ASTRA-RL decomposes RL-based red-teaming into five pieces. Each item below links **directly to source and examples**.

### 1) Problem — *How models run/interact*

Handles loading models/tokenizers, performing **one rollout step** (attacker/target generation), computing **log-probs** (for DPO/PPO, etc.), advancing the conversation state, and defining a **reward**.

* Source: [core/problem.py](../../src/astra_rl/core/problem.py)
* Example reward (ASTPrompter): [methods/ast\_problem.py](../../src/astra_rl/methods/ast_problem.py)
* Full HF adaptor: [ext/transformers/hf\_ast\_problem.py](../../src/astra_rl/ext/transformers/hf_ast_problem.py)
* End-to-end example: [examples/GPT2\_v\_GPT2/ast\_basic\_1.py](../../examples/GPT2_v_GPT2/ast_basic_1.py)
* Guide: [Problem Customization](customize_training/problems.md)

### 2) Environment — *How data is collected*

Defines how attacker–target interactions are generated and structured for training/eval (e.g., **single-path vs. tree** rollouts), what per-step data is stored, and what the solver receives.

* Base: [core/environment.py](../../src/astra_rl/core/environment.py)
* ASTPrompter rollout (`ASTEnvironment`): [methods/ast\_problem.py](../../src/astra_rl/methods/ast_problem.py)
* Guide: [Environment Customization](customize_training/environments.md)

### 3) Moderators — *How we define/measure harm*

Scores target generations (scalar harm). Instantiate in your **Problem** for seamless use.

* Base: [core/moderator.py](../../src/astra_rl/core/moderator.py)
* Detoxify: [moderators/detoxify.py](../../src/astra_rl/moderators/detoxify.py)
* Llama Guard 3: [moderators/llamaGuard.py](../../src/astra_rl/moderators/llamaGuard.py)
* Guide: [Moderator Customization](customize_training/moderators.md)

### 4) Solvers (Algorithms) — *How the attacker learns*

Consume rollout graphs, **flatten** them to per-sample steps, **collate** batches, and compute the **training loss** (plus logs).

* Base: [core/algorithm.py](../../src/astra_rl/core/algorithm.py)
* Examples (DPO/IPO/PPO): [algorithms/](../../src/astra_rl/algorithms)
* Guide: [Solver Customization](customize_training/solvers.md)

### 5) Trainer — *How the training loop runs*

Orchestrates the main loop, hyperparameters, optimizer, eval cadence, and checkpointing.

* Base: [training/trainer.py](../../src/astra_rl/training/trainer.py)
* HF-friendly trainer & config: [ext/transformers/hf\_ast\_problem.py](../../src/astra_rl/ext/transformers/hf_ast_problem.py)
* Guide: [Trainer Customization](customize_training/trainers.md)

---

## 4. Quick Start

### 4.1 Train an RL-Based Attacker

Start here to train with supported moderators, algorithms, and HF attackers/defenders:
**[Quick Start Training](quick_start_training.md)**

**Supported out-of-the-box**

| Component                        | Options                                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Training Algorithms              | PPO, DPO, IPO                                                                                                          |
| Moderators                       | [Llama-Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B), [Detoxify](https://github.com/unitaryai/detoxify) |
| Problem Formulations (published) | [ASTPrompter](https://arxiv.org/abs/2407.09447), [RL – Perez](https://aclanthology.org/2022.emnlp-main.225/)           |

Want to go beyond the defaults? ASTRA-RL is **modular**—swap what you need and reuse everything else. We recommend starting with the quick start to learn the overall flow; it links directly to the relevant **customization guides** when you diverge.

### 4.2 Evaluate a Target with a Pre-Trained Attacker

Jump straight to evaluation (single-path dev/test rollouts, metric aggregation):
**[Quick Start Evaluation](quick_start_evaluation.md)**

---

## 5. References

Casper, S., Lin, J., Kwon, J., Culp, G., & Hadfield-Menell, D. (2023). *Explore, establish, exploit: Red teaming language models from scratch*. arXiv:2306.09442.
Deng, M., Wang, J., Hsieh, C.-P., et al. (2022). *RLPrompt: Optimizing discrete text prompts with reinforcement learning*. EMNLP.
Wichers, N., Denison, C., & Beirami, A. (2024). *Gradient-based language model red teaming*. EACL.
Yu, J., Lin, X., Yu, Z., & Xing, X. (2023). *GPTFuzzer: Red teaming large language models with auto-generated jailbreak prompts*. arXiv:2309.10253.
Zou, A., Wang, Z., Carlini, N., et al. (2023). *Universal and transferable adversarial attacks on aligned language models*. arXiv:2307.15043.
