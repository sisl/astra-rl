# How to Customize the Problem (HF or non-HF)

**Problems** encapsulate *models + tokenization + rollout + log-probabilities + rewards*. Environments call your Problem to:

* sample attacker/target continuations,
* compute log-probs for learning objectives (e.g., DPO/IPO/PPO),
* advance the state,
* acsess attacker parameters,
* compute rewards (scalar or per-step).

Most users can subclass `ASTProblem` (text) or use the HF convenience `HFASTProblem`. If you’re integrating a custom or non-HF model, implement the same small API.

---

## Table of Contents

1. [What Problems Do](#1-what-problems-do)
2. [Built-in / Convenience Problems](#2-built-in--convenience-problems)
3. [Ways to Customize](#3-ways-to-customize)

   * [3.1 Fast path: subclass `HFASTProblem` (HF)](#31-fast-path-subclass-hfastproblem-hf)
   * [3.2 Full control: subclass `ASTProblem` or `Problem`](#32-full-control-subclass-astproblem-or-problem)
4. [Required Interface](#4-required-interface)

   * [4.1 Methods & gradient expectations](#41-methods--gradient-expectations)
   * [4.2 Problem helpers](#42-problem-helpers)
5. [Best Practices & Sanity Checks](#5-best-practices--sanity-checks)
6. [How-Tos](#6-how-tos)

   * [6.1 Minimal HF subclass (`HFASTProblem`)](#61-minimal-hf-subclass-hfastproblem)
   * [6.2 Non-HF custom `Problem` skeleton](#62-non-hf-custom-problem-skeleton)
   * [6.3 Designing rewards](#64-designing-rewards)
   * [6.4 Implementing `advance(...)`](#65-implementing-advance)
   * [6.5 Saving models (HF & non-HF)](#66-saving-models-hf--non-hf)
7. [Plug into Environment / Solver](#7-plug-into-environment--solver)
8. [Debug Checklist](#8-debug-checklist)

---

## 1. What Problems Do

A `Problem` is the bridge between abstract rollouts and concrete model calls. It must:

* **Generate** next utterances for attacker/target,
* **Score** continuations via log-probs,
* **Compute** rewards used by solvers,
* **Advance** the conversation state,
* **Expose** trainable parameters (usually the attacker).

---

## 2. Built-in / Convenience Problems

* **`ASTProblem`** — text-first base with a default `advance()` and a reference reward that combines likelihood and moderator scores (ASTPrompter reward).
* **`HFASTProblem`** — Hugging Face adaptor that subclasses ASTProblem and adds tokenization, generation, and log-prob computation for any HF model.

Use these as templates; override or subclass to fit your needs.

---

## 3. Ways to Customize

### 3.1 Fast path: subclass `HFASTProblem` (HF)

Keep HF models/tokenizers but override specifics (generation kwargs, reward mix, truncation rules, etc.). Minimal code, strong defaults.

### 3.2 Full control: subclass `ASTProblem` or `Problem`

* `ASTProblem` if you’re still doing text but want to own rollout/log-prob/reward details.
* `Problem` for non-text or non-HF stacks—define tokenization/encoding and model calls yourself.

---

## 4. Required Interface

### 4.1 Methods & gradient expectations

Implement **batched** methods (lists in, tensors/lists out, index-aligned):

```python
rollout_prompt_with_attacker(prompts: Sequence[str]) -> Sequence[str]
rollout_prompt_with_target  (prompts: Sequence[str]) -> Sequence[str]

get_attacker_logprobs(contexts: Sequence[str], continuations: Sequence[str]) -> torch.Tensor  # requires grad
get_target_logprobs  (contexts: Sequence[str], continuations: Sequence[str]) -> torch.Tensor  # no grad
get_baseline_logprobs(contexts: Sequence[str], continuations: Sequence[str]) -> torch.Tensor  # no grad

parameters() -> Iterator[torch.nn.Parameter]   # usually attacker params
advance(context: str, attack: str, response: str) -> str # return the next state (i.e. updated conversation context)
reward(contexts, attacks, responses) -> Sequence[float]
```

**Gradients:** only `get_attacker_logprobs` must return a tensor with `requires_grad=True`. Target/baseline should be computed under `torch.no_grad()` (return tensors detached from graphs) to save memory.

### 4.2 Problem helpers

`Problem` provides `_get_*_and_validate(...)` and `_rollout_*_and_validate(...)` utilities that assert shapes and (for attacker) gradient presence. Solvers in this repo call these versions.

---

## 5. Best Practices & Sanity Checks

* **Vectorize everything.** Batch tokenizer/model calls; avoid per-item loops.
* **Mask correctly.** Compute `log P(continuation | context)` by **summing only continuation token log-probs**. *TODO double check this
* **Padding & truncation.** For causal LMs, prefer **left padding** and set `pad_token_id = eos_token_id`. Truncate context to fit `max_ctx - max_new_tokens`.
* **Tokenizer alignment.** Each model (attacker/target/baseline) should encode/decode with its **own** tokenizer.
* **Determinism.** Accept a `seed` from the environment; keep generation settings explicit.
* **Performance.** Use `no_grad` for target/baseline; keep tensors on the correct device.
* **Scale rewards.** Bound/normalize to stabilize PPO-style updates.

---

## 6. How-Tos

### 6.1 Minimal HF subclass (`HFASTProblem`)

```python
from astra_rl.methods.ast_problem import ASTProblem
from astra_rl.core.moderator import Moderator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Sequence, Iterator

class MyHFProblem(ASTProblem):
    def __init__(self, attacker_id: str, target_id: str, baseline_id: str | None,
                 moderator: Moderator[str, str], device: str = "cuda"):
        super().__init__(moderator)
        self.device = device

        self.attacker = AutoModelForCausalLM.from_pretrained(attacker_id).to(device)
        self.a_tok    = AutoTokenizer.from_pretrained(attacker_id)

        self.target   = AutoModelForCausalLM.from_pretrained(target_id).to(device)
        self.t_tok    = AutoTokenizer.from_pretrained(target_id)

        self.baseline = (AutoModelForCausalLM.from_pretrained(baseline_id).to(device)
                         if baseline_id is not None else self.target)
        self.b_tok    = (AutoTokenizer.from_pretrained(baseline_id)
                         if baseline_id is not None else self.t_tok)

        for tok in (self.a_tok, self.t_tok, self.b_tok):
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id
            tok.padding_side = "left"  # set on the tokenizer object

        # Optional: cache max ctx
        self.max_ctx = getattr(self.attacker.config, "n_positions",
                        getattr(self.attacker.config, "max_position_embeddings", 1024))

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.attacker.parameters()

    # ---- generation ----
    def rollout_prompt_with_attacker(self, prompts: Sequence[str]) -> Sequence[str]:
        return self._rollout(self.attacker, self.a_tok, prompts)

    def rollout_prompt_with_target(self, prompts: Sequence[str]) -> Sequence[str]:
        return self._rollout(self.target, self.t_tok, prompts)

    def _rollout(self, model, tok, prompts):
        max_new = 32
        enc = tok(prompts, padding=True, return_tensors="pt", truncation=True,
                  max_length=self.max_ctx - max_new).to(self.device)
        with torch.no_grad():
            out = model.generate(**enc, pad_token_id=tok.eos_token_id,
                                 max_new_tokens=max_new, do_sample=True,
                                 top_p=0.9, top_k=50, temperature=1.0)
        texts = tok.batch_decode(out, skip_special_tokens=True)
        return [full[len(p):] for full, p in zip(texts, prompts)]

    # ---- logprobs ----
    def get_attacker_logprobs(self, ctx, cont):
        return self._get_logprobs(self.attacker, self.a_tok, ctx, cont, requires_grad=True)

    def get_target_logprobs(self, ctx, cont):
        with torch.no_grad():
            return self._get_logprobs(self.target, self.t_tok, ctx, cont, requires_grad=False)

    def get_baseline_logprobs(self, ctx, cont):
        with torch.no_grad():
            return self._get_logprobs(self.baseline, self.b_tok, ctx, cont, requires_grad=False)

    def _get_logprobs(self, model, tok, ctx, cont, requires_grad=False):
        # Encode separately, build mask to sum only continuation tokens
        ctx_ids = tok(ctx).input_ids
        cont_ids = tok(cont).input_ids
        mask = [[False]*len(c) + [True]*len(r) for c, r in zip(ctx_ids, cont_ids)]
        combo = [c + r for c, r in zip(ctx_ids, cont_ids)]
        L = max(len(x) for x in combo)
        pad = tok.eos_token_id

        combo = [x + [pad]*(L-len(x)) for x in combo]
        mask  = [m + [False]*(L-len(m)) for m in mask]
        attn  = [[1]*len(m) + [0]*(L-len(m)) for m in mask]

        combo = torch.tensor(combo, device=self.device)
        attn  = torch.tensor(attn,  device=self.device)
        mask  = torch.tensor(mask,  device=self.device)

        logits = model(input_ids=combo, attention_mask=attn).logits[:, :-1].log_softmax(-1)
        token_lp = logits.gather(-1, combo[:, 1:].unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp.masked_fill(~mask[:, 1:], 0.0)
        return token_lp.sum(dim=-1)  # shape [B]
```

**Notes**

* Prefer setting `tokenizer.padding_side = "left"` on the object (don’t pass `padding_side` to `__call__`).
* When preparing **input contexts** you’ll later concatenate with generated text, leave `add_special_tokens=False` to avoid BOS/SEP drift.
* Keep `max_ctx - max_new_tokens` headroom to prevent device-side indexing errors.

### 6.2 Non-HF custom `Problem` skeleton

```python
class MyProblem(Problem[str, str]):
    def __init__(self, moderator, attacker_model, target_model, baseline_model, device="cuda"):
        super().__init__(moderator)
        self.device = device
        self.attacker = attacker_model.to(device)
        self.target   = target_model.to(device)
        self.baseline = baseline_model.to(device)

        # TODO: load the attacker, target and baseline tokenizers
        # TODO: set your padding tokenins for each tokenizer
        # TODO: set your model's usable max sequence length (e.g GPT-2: 1024)

    def rollout_prompt_with_attacker(self, prompts):
        # TODO: your generator over token ids/text → list[str] continuations
        ...

    def rollout_prompt_with_target(self, prompts):
        ...

    def get_attacker_logprobs(self, ctx, cont):
        # Return sum log P(cont | ctx) per example; tensor requires grad
        return self._logprobs(self.attacker, ctx, cont, requires_grad=True)

    def get_target_logprobs(self, ctx, cont):
        with torch.no_grad():
            return self._logprobs(self.target, ctx, cont, requires_grad=False)

    def get_baseline_logprobs(self, ctx, cont):
        with torch.no_grad():
            return self._logprobs(self.baseline, ctx, cont, requires_grad=False)

    def _logprobs(self, model, ctx, cont, requires_grad):
        # Implement your own encode/combine/mask logic (not HF)
        # 1) encode ctx, cont → id tensors
        # 2) build attention + continuation mask
        # 3) forward model → logits → log_softmax
        # 4) gather per-token logprobs, mask out context, sum over continuation
        ...

    def advance(self, context, attack, response):
        # Conversation concatenation or your custom state transition
        return context + attack + response

    def parameters(self):
        return self.attacker.parameters()

    def reward(self, contexts, attacks, responses):
        # calculate your custom reward 
        return r
```

### 6.3 Designing rewards

Common patterns (return one float per sample):

* **harm-driven:** use moderator-generated scores for defender harm as a key component of the reward
* **Preference methods (DPO/IPO/ORPO):** may not use rewards directly; rely on log-prob differences.
* **Tips:** bound/clip; normalize across a batch; document “higher is worse” vs “higher is better”.

### 6.4 implementing `advance(...)`

Default text setting is simple concatenation (below) but you can customize how the next state is created.

```python
def advance(self, context, attack, response):
    return context + attack + response
```


### 6.5 Saving models (HF & non-HF)

* **HF:** `model.save_pretrained(path)` and `tokenizer.save_pretrained(path)`.
* **Non-HF:** `torch.save(model.state_dict(), path)` and a small loader util. Ensure your trainer saves anything else your algorithm needs (e.g., optimizer/scheduler state).

---

## 7. Plug into Environment / Solver

Your problem will be passed to the 'Environment' and 'Solver'. The 'Trainer' will have acsess to the problem through the environment (env.problem).
```python
problem = MyHFProblem("gpt2", "gpt2", "gpt2", DetoxifyModerator(), device="cuda")
env     = ASTEnvironment(problem, PROMPTS, tree_width=2, tree_depth=3)
solver  = DPO(problem, beta=0.1)
trainer = Trainer(config=config, environment=env, algorithm=solver)
trainer.train()
```

---

## 8. Debug Checklist

* **Batching:** input list lengths match; outputs align (`[B]` tensors).
* **Gradients:** attacker log-probs require grad; target/baseline under `no_grad`.
* **Masking:** only continuation tokens contribute to `log P(cont | ctx)`.
* **Context window:** `len(ctx_tokens) + max_new_tokens ≤ max_ctx`.
* **Tokenizer differences:** never cross-decode; keep model/tokenizer pairs.
* **Device/type:** tensors on right device/dtype; `pad_token_id` set.
* **Numerics:** watch for `nan/inf`; clip/normalize rewards.
* **Repro:** fixed seeds for rollout sampling and generation settings.

