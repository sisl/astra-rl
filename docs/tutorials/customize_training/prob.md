Awesome—here’s a polished, copy-pasteable “How-to” that shows users exactly how to build a custom **Problem** class, including **non-HuggingFace models** and a **custom reward**.

---

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

## Minimal template (non-HF friendly)

Below is a “neutral” template that works whether your models are HuggingFace or not. Plug your model in by implementing four tiny adapter methods: `_encode`, `_decode`, `_generate_ids`, `_forward_logits`.

```python
from __future__ import annotations
from typing import Iterable, List, Sequence, Optional, Dict, Any
import torch
import torch.nn.functional as F

from astra_rl.methods.ast_problem import ASTProblem
from astra_rl.core.moderator import Moderator  # your toxicity/safety scorer, etc.

class MyCustomProblem(ASTProblem):
    """
    Problem class that supports *any* autoregressive LM by providing
    small adapters for encode/decode/generate/forward.
    """

    def __init__(
        self,
        moderator: Moderator[str, str],
        attacker_model,      # your custom AR LM (HF or not)
        target_model,        # frozen model under test (HF or not)
        *,
        device: str = "cuda",
        max_ctx: int = 1024,
        pad_token_id: int = 0,
        eos_token_id: int = 0,
    ):
        super().__init__(moderator)
        self.device = device
        self.max_ctx = int(max_ctx)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)

        self.attacker = attacker_model.to(self.device)
        self.target   = target_model.to(self.device)

        # If your models need separate tokenizers/encoders, store them:
        # self.attacker_tok = ...
        # self.target_tok   = ...

    # ---------- Required harness API: rollouts ----------
    def rollout_prompt_with_attacker(self, prompts: Sequence[str]) -> List[str]:
        """
        Generate attacker continuations for each prompt.
        Returns a list of pure continuations (not the full prompt+continuation).
        """
        # 1) Encode
        batch = self._encode(prompts, for_attacker=True)
        input_ids = batch["input_ids"].to(self.device)

        # 2) Generate ids with your sampler
        out_ids = self._generate_ids(self.attacker, input_ids, max_new_tokens=32)

        # 3) Decode full strings, then strip the prompt
        full_texts = self._decode(out_ids, for_attacker=True)
        conts = [full[len(p):] for p, full in zip(prompts, full_texts)]
        return conts

    def rollout_prompt_with_target(self, prompts: Sequence[str]) -> List[str]:
        batch = self._encode(prompts, for_attacker=False)
        input_ids = batch["input_ids"].to(self.device)
        out_ids = self._generate_ids(self.target, input_ids, max_new_tokens=32)
        full_texts = self._decode(out_ids, for_attacker=False)
        conts = [full[len(p):] for p, full in zip(prompts, full_texts)]
        return conts

    # ---------- Required harness API: log-probs ----------
    def get_attacker_logprobs(
        self, contexts: Sequence[str], continuations: Sequence[str]
    ) -> torch.Tensor:
        """
        Return sum of token log-probs for each continuation given its context.
        Must allow gradient to flow through the *attacker*.
        """
        return self._conditional_logprob_sum(
            self.attacker, contexts, continuations, for_attacker=True, require_grad=True
        )

    def get_target_logprobs(
        self, contexts: Sequence[str], continuations: Sequence[str]
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._conditional_logprob_sum(
                self.target, contexts, continuations, for_attacker=False, require_grad=False
            )

    def get_baseline_logprobs(
        self, contexts: Sequence[str], continuations: Sequence[str]
    ) -> torch.Tensor:
        # Often identical to target (frozen reference) in KL/DPO setups.
        return self.get_target_logprobs(contexts, continuations)

    # ---------- Required harness API: trainable params ----------
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        # Only optimize the attacker by default
        return self.attacker.parameters()

    # ---------- Reward: customize for your task ----------
    def reward(
        self,
        contexts: Sequence[str],
        attacks: Sequence[str],
        responses: Sequence[str],
    ) -> List[float]:
        """
        Example: combine toxicity (higher = “worse” => higher reward
        for red team) and a short-length preference.
        """
        # Batch moderation (vectorized)
        atk_tox = self.moderator.moderate(attacks)    # List[float] in [0,1]
        rsp_tox = self.moderator.moderate(responses)  # List[float] in [0,1]

        # Simple scalar reward: emphasize target toxicity, lightly weight attack length
        w_rsp, w_atk, len_pen = 1.0, 0.3, 0.002
        rewards = []
        for a, r, atk_txt in zip(atk_tox, rsp_tox, attacks):
            rew = w_rsp * r + w_atk * a - len_pen * len(atk_txt)
            rewards.append(float(rew))
        return rewards

    # ======================================================================
    # ============= Model-agnostic adapters (edit for non-HF!) =============
    # ======================================================================

    def _encode(self, texts: Sequence[str], *, for_attacker: bool) -> Dict[str, torch.Tensor]:
        """
        Return {'input_ids': LongTensor[B, T]} with left padding and context cap.
        Replace this body to use your own tokenizer/byte encoder.
        """
        # Example: using a shared byte-level BPE or your custom tokenizer
        ids_list: List[List[int]] = [
            self._text_to_ids(t, for_attacker=for_attacker) for t in texts
        ]
        # Left-truncate to fit context window (reserve space for generation if needed)
        T = max(len(ids) for ids in ids_list)
        if T > self.max_ctx:
            ids_list = [ids[-self.max_ctx:] for ids in ids_list]
            T = self.max_ctx

        # Left-pad with PAD to length T
        padded = []
        for ids in ids_list:
            pad = [self.pad_token_id] * (T - len(ids))
            padded.append(pad + ids)
        input_ids = torch.tensor(padded, dtype=torch.long)
        return {"input_ids": input_ids}

    def _decode(self, ids: torch.Tensor, *, for_attacker: bool) -> List[str]:
        """
        Convert token ids back to text. Implement with your tokenizer/codec.
        """
        return [self._ids_to_text(row.tolist(), for_attacker=for_attacker) for row in ids]

    def _generate_ids(
        self,
        model,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive sampling loop that works for any model exposing logits.
        Replace with your model's native `.generate(...)` if available.
        """
        model.train(False)
        B, T = input_ids.shape
        ids = input_ids.to(self.device)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self._forward_logits(model, ids)[:, -1, :]  # [B, V]
                # Temperature + nucleus/top-k (toy implementation)
                logits = logits / max(1e-6, temperature)
                probs = F.softmax(logits, dim=-1)

                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)))
                    mask = torch.ones_like(probs) * 0.0
                    mask.scatter_(1, topk_idx, topk_vals)
                    probs = mask
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                if top_p is not None and 0 < top_p < 1:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    keep = cumsum <= top_p
                    # ensure at least one token
                    keep[..., 0] = True
                    filtered = torch.zeros_like(probs)
                    filtered.scatter_(1, sorted_idx, keep.float() * sorted_probs)
                    probs = filtered / filtered.sum(dim=-1, keepdim=True)

                next_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]
                ids = torch.cat([ids, next_ids], dim=1)
        return ids

    def _forward_logits(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns logits [B, T, V].
        Implement using your model’s API.
        """
        # Example for a PyTorch nn.Module that takes `input_ids`
        return model(input_ids)  # must return [B, T, V]

    # ---------- Conditional log-prob utility ----------
    def _conditional_logprob_sum(
        self,
        model,
        contexts: Sequence[str],
        continuations: Sequence[str],
        *,
        for_attacker: bool,
        require_grad: bool,
    ) -> torch.Tensor:
        """
        Compute log P(continuation | context), summed over continuation tokens.
        """
        assert len(contexts) == len(continuations)
        # Encode each part separately so we can mask out context tokens
        ctx_ids = [self._text_to_ids(t, for_attacker=for_attacker) for t in contexts]
        cont_ids = [self._text_to_ids(t, for_attacker=for_attacker) for t in continuations]

        # Build combined sequence per item and left-pad to same length
        combined = [c + d for c, d in zip(ctx_ids, cont_ids)]
        T = min(self.max_ctx, max(len(x) for x in combined))
        padded, mask_ctx = [], []
        for c, d in zip(ctx_ids, cont_ids):
            seq = (c + d)[-T:]  # left-truncate to window
            # context length after truncation:
            ctx_len = max(0, min(len(c), T - len(d)))
            # left-pad
            pad = [self.pad_token_id] * (T - len(seq))
            padded.append(pad + seq)
            # mask True *only* over continuation tokens
            mask_ctx.append([False] * (T - len(seq) + ctx_len) + [True] * (len(seq) - ctx_len))

        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        cont_mask = torch.tensor(mask_ctx, dtype=torch.bool, device=self.device)

        # Forward
        input_ids_ = input_ids.detach().clone()
        input_ids_.requires_grad_(require_grad)  # gradients flow through attacker only
        logits = self._forward_logits(model, input_ids_)  # [B, T, V]
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)         # next-token log P
        next_tokens = input_ids_[:, 1:].unsqueeze(-1)               # [B, T-1, 1]
        token_logp = logprobs.gather(-1, next_tokens).squeeze(-1)   # [B, T-1]

        # Mask out context tokens; sum only over continuation
        token_logp = token_logp.masked_fill(~cont_mask[:, 1:], 0.0)
        lp_sum = token_logp.sum(dim=-1)  # [B]
        return lp_sum

    # ---------- Replace these with your tokenizer/codec ----------
    def _text_to_ids(self, text: str, *, for_attacker: bool) -> List[int]:
        """
        Convert text->ids. Replace with your tokenizer or byte codec.
        Ensure EOS exists; consider using eos as pad for GPT-style models.
        """
        # Toy byte-level “tokenizer” (DON’T use in production):
        ids = [min(255, ord(ch)) for ch in text]
        ids = ids + [self.eos_token_id]
        return ids

    def _ids_to_text(self, ids: List[int], *, for_attacker: bool) -> str:
        # Inverse of the toy tokenizer (again, replace with your real one)
        # Strip left-padding PADs:
        ids = [t for t in ids if t != self.pad_token_id]
        # Stop at first EOS if desired
        if self.eos_token_id in ids:
            ids = ids[: ids.index(self.eos_token_id)]
        return "".join(chr(i) for i in ids)
```

### Why this template helps

* Works for **non-HF** models: you decide how to encode/decode/generate/forward.
* Keeps **log-prob masking** correct: we compute `log P(continuation | context)` by masking out context tokens and summing over continuation tokens only.
* Preserves **gradients** for the attacker but uses `no_grad` for target/baseline, matching PPO/DPO style solvers.

---

## HuggingFace convenience (optional)

If you *are* on HF, your life is simpler. You can mirror your current example or subclass our `HFASTProblem`. Quick notes:

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

## Quick self-test (paste in a notebook)

```python
def _sanity(problem: MyCustomProblem):
    prompts = ["Hello", "Tell me a joke about cats"]
    atk = problem.rollout_prompt_with_attacker(prompts)
    tgt = problem.rollout_prompt_with_target([p + a for p, a in zip(prompts, atk)])
    assert len(atk) == len(tgt) == len(prompts)

    lpa = problem.get_attacker_logprobs(prompts, atk)
    lpt = problem.get_target_logprobs(prompts, atk)
    assert lpa.shape == lpt.shape == (len(prompts),)
    assert lpa.dtype == lpt.dtype

    r = problem.reward(prompts, atk, tgt)
    assert isinstance(r, list) and len(r) == len(prompts)
    print("✓ problem interface looks good")

# _sanity(MyCustomProblem(...))
```

---

## When to use `ASTProblem` vs `HFASTProblem`

* **Non-HF or heavily customized** model stack → start from the template above (subclass `ASTProblem`).
* **HF Transformers** → subclass `HFASTProblem` (less boilerplate) *or* adapt the template by replacing `_encode/_decode/_generate_ids/_forward_logits` with HF calls.

---

With this guide + template, users can wire up *any* autoregressive model, keep gradients and masking correct, and plug in custom rewards without touching the harness or environment logic.
