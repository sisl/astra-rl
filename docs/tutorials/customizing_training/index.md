# Customize Training

To go beyond the supported training classes presented in the [quick start training guide](../quick_start_training.md), please see the following resources:

### 1) System — *How models run/interact*

Handles loading models/tokenizers, performing **one rollout step** (tester/target generation), computing **log-probs** (for DPO/PPO, etc.), advancing the conversation state, and defining a **reward**.

* Guide: [System Customization](problems.md)

### 2) Sampler — *How data is collected*

Defines how tester–target interactions are generated and structured for training/eval (e.g., **single-path vs. tree** rollouts), what per-step data is stored, and what the solver receives.

* Guide: [Sampler Customization](environments.md)

### 3) Scorers — *How we define/measure harm*

Scores target generations (scalar harm). Instantiate in your **System** for seamless use.

* Guide: [Scorer Customization](moderators.md)

### 4) Solvers (Algorithms) — *How the tester learns*

Consume rollout graphs, **flatten** them to per-sample steps, **collate** batches, and compute the **training loss** (plus logs).

* Guide: [Solver Customization](solvers.md)

### 5) Trainer — *How the training loop runs*

Orchestrates the main loop, hyperparameters, optimizer, eval cadence, and checkpointing.

* Guide: [Trainer Customization](trainers.md)
