# How to Create a Custom Problem Class (HF or non-HF)

A **Problem** encapsulates *models + tokenization + rollout + log-probabilities + rewards*. Environments call your Problem to:

* sample attacker/target continuations,
* compute log-probs for losses (e.g., PPO/DPO),
* compute rewards (scalar or per-step).

Most users can subclass `ASTProblem` (or `HFASTProblem` for HF). If you’re integrating a custom / non-HF model, you’ll implement the same small set of methods.

---

## What you must implement (API contract)

Your subclass must provide batched implementations (lists in, lists/tensors out), aligned by index:

* `rollout_prompt_with_attacker(prompts: Sequence[str]) -> Sequence[str]`
* `rollout_prompt_with_target(prompts: Sequence[str]) -> Sequence[str]`
* `get_attacker_logprobs(contexts: Sequence[str], continuations: Sequence[str]) -> torch.Tensor`
  *(sum of token log-probs for each continuation conditioned on its context; shape `[B]`)*
* `get_target_logprobs(contexts, continuations) -> torch.Tensor` *(no grad)*
* `get_baseline_logprobs(contexts, continuations) -> torch.Tensor` *(no grad; often same as target/reference)*
* `parameters() -> Iterable[torch.nn.Parameter]` *(usually attacker’s params only)*

> **Vectorize!** Always pass lists (even of length 1) and do batched compute inside. This is faster and simpler.

---
## TODO: Allie add details on how to create a custom problem class that...
* Works for **non-HF** models: user decides how to encode/decode/generate/forward.
* Keeps **log-prob masking** correct: compute `log P(continuation | context)` by masking out context tokens and summing over continuation tokens only.
* Preserves **gradients** for the attacker but uses `no_grad` for target/baseline, matching PPO/DPO style solvers.

---

## HuggingFace convenience

If you *are* using HF, you can subclass our `HFASTProblem`. Quick notes:

* Use **left padding + EOS as PAD** for causal LMs to keep the *rightmost* tokens aligned.
* Keep `add_special_tokens=False` when encoding prompts that you’ll later concatenate with generated text—this avoids inserting extra BOS/SEP that break index alignment.
* Respect model context: truncate `context` to `max_ctx - max_new_tokens` before generation to avoid device-side index errors.

---

## Designing a custom reward

Your `reward(contexts, attacks, responses)` returns a list of floats aligned with the batch. Common patterns:

* **Toxicity-driven** (maximize target toxicity):
  `reward = w1 * tox(response) + w2 * tox(attack) - λ * length(attack)`
* **Safety violations** from a separate policy/guardrail model.
* **Preference pairs**: when using DPO/IPO/ORPO, you’ll pass log-probs for preferred/dispreferred candidates to the solver; `reward` may be unused.
* **Task-specific**: factuality, jailbreak success, refusal suppression, etc.

Tip: keep rewards *bounded* (e.g., clip to \[-1, 1]) to stabilize PPO-style updates.

---

## Practical gotchas

* **Return log-probs, not probs.** Only exponentiate if some downstream code explicitly asks for probabilities.
* **Gradient locality.** Let gradient flow through `get_attacker_logprobs`. Use `torch.no_grad()` for target/baseline calls.
* **Batch always.** Avoid per-item loops calling the model. Build padded batches.
* **Context windows.** Enforce `T ≤ max_ctx`. For GPT-like models, left-truncate context and reserve space for the continuation.
* **Tokenizer mismatch.** If attacker/target tokenizers differ, do *not* cross-decode. Each model should encode/decode its own texts.
* **Determinism.** Accept a random seed (or RNG) in your env for reproducible rollouts.

---

## When to use `ASTProblem` vs `HFASTProblem`

* **Non-HF or heavily customized** model stack → subclass `ASTProblem`.
* **HF Transformers** → subclass `HFASTProblem` (less boilerplate) 

---
