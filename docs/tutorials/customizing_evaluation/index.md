# Customize Evaluation

**RL-based red-teaming** trains an *attacker* via reinforcement learning so it generates test cases that are likely to elicit unsafe outputs from a *target* model. Attackers probe targets through multi-turn conversations and try to surface failure modes (unsafe/harmful outputs).

ASTRA-RL supports both **training** attackers and **evaluating** targets. In evaluation mode we repeatedly test targets using a pre-trained attacker and collect metrics that describe how, when, and how often a target fails.

!!! note "What counts as a successful attack?"
    An *attack is successful* when the target model produces an unsafe or harmful utterance according to the configured moderator (e.g., Detoxify, LlamaGuard). Red-teaming’s purpose is to discover as many such failure modes as possible so they can be analyzed and mitigated.

---

## Quick links

- To run a red-team evaluation using a trained attacker, follow the quick guide: **[Quick Start: Evaluation](../quick_start_evaluation.md)**.  
  This shows how to use the supported evaluation problems (`HFEvaluationProblem` for HF models and `GPT2EvaluationProblem` for GPT-2) and the default evaluator (`ASTEvaluator`).

- If you need to support a custom model or tokenizer, see **[Evaluation Problem Customization](evaluation_problems.md)**.

- If you want to collect different per-turn or aggregated metrics, or change how metrics are computed/serialized, see **[Evaluator Customization](evaluators.md)**.

---

## Short workflow

1. Train an attacker (see **Quick Start: Training**).  
2. Point evaluation at your attacker checkpoint.  
3. Run `ASTEvaluator` over a set of held-out prompts (never used at training time).  
4. Inspect per-turn logs and aggregated metrics (JSON output) to find failure modes.

---

## Tips

- Use a moderator that matches your safety criteria (e.g., toxicity vs. policy violations).  
- Keep evaluation prompts *out-of-sample* to avoid reporting overfit behavior.  

