# Tutorials

Welcome to ASTRA-RL! This section provides step-by-step guides and examples to help you get started with the ASTRA-RL toolbox.

ASTRA-RL is a user-friendly, modular, and customizable toolbox for **Langauge Model Red-Teaming**. It's designed for quickly getting started evaluating models or building your own red-teaming pipelines for training and evaluation.

---

## What is Lanaguage Model Red-Teaming?

**Language model red-teaming** aims to identify and benchmark prompts that elicit harmful or otherwise undesirable behavior from a target language model ([Hardy et al., 2025](https://arxiv.org/abs/2407.09447))[^1]. This surfaces vulnerabilities and guides fine-tuning to reduce harmful outputs.

* **Manual red-teaming:** human annotators craft adversarial prompts—effective but costly and not scalable ([Ganguli et al., 2022](https://arxiv.org/abs/2209.07858))[^2].
* **Automated red-teaming:** generates adversarial prompts at scale. Examples include fuzzing exisiting prompts ([Yu et al., 2023](https://arxiv.org/abs/2309.10253))[^3] or using a combination of targeted search techniques to optimize an effective adversarial suffix ([Zou et al., 2023](https://arxiv.org/abs/2307.15043))[^4]

### RL-based Red-Teaming

A promising direction in automated red-teaming is **reinforcement learning (RL)**. We train a separate **attacker** policy (often an LLM) to maximize a **non-differentiable reward** (e.g., toxicity) computed on the target's response. In short, we **automate** red-teaming by training an attacker to generate test cases that increase the chance of unsafe target outputs.

**Pros**

1. **Fast at inference:** once trained, generating new prompts is quick and inexpensive.
2. **Effective:** prior work (e.g., Perez; Huang; Hardy) shows RL-trained attackers reliably induce harmful behavior.

**Cons**

1. **Mode collapse / low coverage:** may find only a small set of effective patterns ([Casper et al., 2023](https://arxiv.org/abs/2306.09442))[^5].
2. **Unrealistic prompts:** can be disfluent or implausible ([Casper et al., 2023](https://arxiv.org/abs/2306.09442);[Deng et al., 2022](https://arxiv.org/abs/2205.12548))[^5][^6], even with realism terms ([Wichers et al., 2024](https://arxiv.org/abs/2401.16656))[^7].

ASTRA-RL makes **training** an RL attacker and **evaluating** a target with a pre-trained attacker both **quick** and **customizable**.

---

## Key Terminology

* **Target** (a.k.a. *defender*, *model under test*)
  The model being red-teamed. It converses with the attacker; the target's response is scored by a **moderator**, and that score contributes to the attacker's reward.

* **Attacker**
  The policy (often an LLM) that generates utterances intended to elicit harmful responses from the target. Typically initialized from a general LM (e.g., Llama-2) and **updated via RL** to improve effectiveness.

* **Moderator**
  The scoring component (like a reward model). At each step, it returns a **scalar measure of harm** (e.g., toxicity). "Harm" can be defined via existing classifiers (e.g., Llama-Guard 3) or a custom model you provide.

---

## Package Overview

ASTRA-RL decomposes RL-based red-teaming into five pieces. 

### 1) Problem — *How models run/interact*

Handles loading models/tokenizers, performing **one rollout step** (attacker/target generation), computing **log-probs** (for DPO/PPO, etc.), advancing the conversation state, and defining a **reward**.

* Guide: [Problem Customization](customize_training/problems.md)

### 2) Environment — *How data is collected*

Defines how attacker–target interactions are generated and structured for training/eval (e.g., **single-path vs. tree** rollouts), what per-step data is stored, and what the solver receives.

* Guide: [Environment Customization](customize_training/environments.md)

### 3) Moderators — *How we define/measure harm*

Scores target generations (scalar harm). Instantiate in your **Problem** for seamless use.

* Guide: [Moderator Customization](customize_training/moderators.md)

### 4) Solvers (Algorithms) — *How the attacker learns*

Consume rollout graphs, **flatten** them to per-sample steps, **collate** batches, and compute the **training loss** (plus logs).

* Guide: [Solver Customization](customize_training/solvers.md)

### 5) Trainer — *How the training loop runs*

Orchestrates the main loop, hyperparameters, optimizer, eval cadence, and checkpointing.

* Guide: [Trainer Customization](customize_training/trainers.md)

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
| Problem Formulations (published) | [ASTPrompter](https://arxiv.org/abs/2407.09447), [RL – Perez](https://aclanthology.org/2022.emnlp-main.225/)           |

Want to go beyond the defaults? ASTRA-RL is **modular**—swap what you need and reuse everything else. We recommend starting with the quick start to learn the overall flow; it links directly to the relevant **customization guides** when you diverge.

### Evaluate a Target with a Pre-Trained Attacker

Jump straight to evaluation (single-path dev/test rollouts, metric aggregation):
**[Quick Start Evaluation](quick_start_evaluation.md)**

[^1]: Hardy, A., Liu, H., Lange, B., Eddy, D., and Kochenderfer, M.J. *ASTPrompter: Weakly Supervised Automated Language Model Red-Teaming to Identify Low-Perplexity Toxic Prompts.*. [arXiv:2407.09447](https://arxiv.org/abs/2407.09447) (2024)

[^2]: Ganguli, D., Li, K., Shumailov, I., et al. *Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned*. [arXiv:2212.08073](https://arxiv.org/abs/2209.07858) (2022)

[^3]: Yu, J., Lin, X., Yu, Z., & Xing, X. *GPTFuzzer: Red teaming large language models with auto-generated jailbreak prompts*. [arXiv:2309.10253](https://arxiv.org/abs/2309.10253) (2023)

[^4]: Zou, A., Wang, Z., Carlini, N., et al. (2023). *Universal and transferable adversarial attacks on aligned language models*. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) (2023)

[^5]: Casper, S., Lin, J., Kwon, J., Culp, G., & Hadfield-Menell, D. *Explore, establish, exploit: Red teaming language models from scratch*. [arXiv:2306.09442](https://arxiv.org/abs/2306.09442) (2023).

[^6]: Deng, M., Wang, J., Hsieh, C.-P., et al. (2022). *RLPrompt: Optimizing discrete text prompts with reinforcement learning*. [arXiv:2205.12548](https://arxiv.org/abs/2205.12548) (2022)

[^7]: Wichers, N., Denison, C., & Beirami, A. *Gradient-based language model red teaming* [arXiv:2401.16656](https://arxiv.org/abs/2401.16656) (2024)