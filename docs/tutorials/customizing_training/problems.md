# Systems

**Systems** encapsulate *models + tokenization + rollout + log-probabilities + rewards*. Samplers call your System to:

* sample auditor/target continuations,
* compute log-probs for learning objectives (e.g., DPO/IPO/PPO),
* advance the state,
* access auditor parameters,
* compute rewards (scalar or per-step).

Most users can subclass `ASTSystem` (text) or use the HF convenience `HFASTSystem`. If you're integrating a custom or non-HF model, implement the same small API.

---

## 1. What Systems Do

A `System` is the bridge between abstract rollouts and concrete model calls. It must:

* **Generate** next utterances for auditor/target,
* **Score** continuations via log-probs,
* **Compute** rewards used by solvers,
* **Advance** the conversation state,
* **Expose** trainable parameters (usually the auditor).

---

## 2. Built-in / Convenience Systems

* **`ASTSystem`** — text-first base with a default `advance()` and a reference reward that combines likelihood and scorer scores (ASTPrompter reward).
* **`HFASTSystem`** — Hugging Face adaptor that subclasses ASTSystem and adds tokenization, generation, and log-prob computation for any HF model that does not have a fixed length (eg. not GPT2, but works for other models like llama models).

Use these as templates; override or subclass to fit your needs.

---

## 3. Ways to Customize

### 3.1 Fast path: subclass `HFASTSystem` (HF)

Keep HF models/tokenizers but override specifics (generation kwargs, reward mix, truncation rules, etc.). Minimal code, strong defaults.

### 3.2 Full control: subclass `ASTSystem` or `System`

* `ASTSystem` if you're still doing text but want to own rollout/log-prob/reward details.
* `System` for non-text or non-HF stacks—define tokenization/encoding and model calls yourself.

---

## 4. Required Interface

### 4.1 Methods & gradient expectations

Every System must implement the following **batched** methods (lists in, tensors/lists out, index-aligned):

```python
rollout_prompt_with_auditor(prompts: Sequence[str]) -> Sequence[str]
rollout_prompt_with_target  (prompts: Sequence[str]) -> Sequence[str]

get_auditor_logprobs(contexts: Sequence[str], continuations: Sequence[str]) -> torch.Tensor  # requires grad
get_target_logprobs  (contexts: Sequence[str], continuations: Sequence[str]) -> torch.Tensor  # no grad
get_baseline_logprobs(contexts: Sequence[str], continuations: Sequence[str]) -> torch.Tensor  # no grad

parameters() -> Iterator[torch.nn.Parameter]   # usually auditor params
advance(context: str, probe: str, response: str) -> str # return the next state (i.e. updated conversation context)
reward(contexts, probes, responses) -> Sequence[float]
```

**Gradients:** only `get_auditor_logprobs` must return a tensor with `requires_grad=True`. Target/baseline should be computed under `torch.no_grad()` (return tensors detached from graphs) to save memory.

### 4.2 System helpers

`System` provides `_get_*_and_validate(...)` and `_rollout_*_and_validate(...)` utilities that assert shapes and (for auditor) gradient presence. Solvers in this repo call these versions.

---

## 5. Best Practices & Sanity Checks

* **Vectorize everything.** Batch tokenizer/model calls; avoid per-item loops.
* **Mask correctly.** Compute `log P(continuation | context)` by **summing only continuation token log-probs**.
* **Padding & truncation.** For causal LMs, prefer **left padding** and set `pad_token_id = eos_token_id`. Truncate context to fit `max_ctx - max_new_tokens`.
* **Tokenizer alignment.** Each model (auditor/target/baseline) should encode/decode with its **own** tokenizer.
* **Determinism.** Accept a `seed` from the sampler; keep generation settings explicit.
* **Performance.** Use `no_grad` for target/baseline; keep tensors on the correct device.
* **Scale rewards.** Bound/normalize to stabilize PPO-style updates.

---

## 6. How-Tos

To create your custom System class, we encourage you to find what existing system is the closest fit to your desired System and subclass from there.

For example, if you are still using huggingface models but want to change how the state advances or how the reward is calculated, you should subclass from the HFASTSystem, define the methods you wish to change, and let the pre-defined methods in HFASTSystem remain.

The following code shows how to subclass from the base System class and where you should implement your custom methods to create the changes you desire.

```python
class MySystem(System[str, str]):
    def __init__(self, scorer, auditor_model, target_model, baseline_model, device="cuda"):
        super().__init__(scorer)
        self.device = device

        # set your auditor, target, and baseline models
        self.auditor = auditor_model.to(device)
        self.target   = target_model.to(device)
        self.baseline = baseline_model.to(device)

        # TODO: load the auditor, target and baseline tokenizers
        # TODO: set your padding tokens for each tokenizer
        # TODO: set your model's usable max sequence length (e.g GPT-2: 1024)

    def rollout_prompt_with_auditor(self, prompts):
        # TODO: your generator over token ids/text → list[str] continuations
        ...

    def rollout_prompt_with_target(self, prompts):
        ...

    def get_auditor_logprobs(self, ctx, cont):
        # Return sum log P(cont | ctx) per example; tensor requires grad
        return self._logprobs(self.auditor, ctx, cont, requires_grad=True)

    def get_target_logprobs(self, ctx, cont):
        with torch.no_grad():
            return self._logprobs(self.target, ctx, cont, requires_grad=False)

    def get_baseline_logprobs(self, ctx, cont):
        with torch.no_grad():
            return self._logprobs(self.baseline, ctx, cont, requires_grad=False)

    def _logprobs(self, model, ctx, cont, requires_grad):
        # Implement your own encode/combine/mask logic
        # 1) encode ctx, cont → id tensors
        # 2) build attention + continuation mask
        # 3) forward model → logits → log_softmax
        # 4) gather per-token logprobs, mask out context, sum over continuation
        ...

    def advance(self, context, probe, response):
        # Conversation concatenation or your custom state transition
        return context + probe + response

    def parameters(self):
        return self.auditor.parameters()

    def reward(self, contexts, probes, responses):
        # calculate your custom reward here!
        # return a scalar value. Note that binary signals are not as helpful for training. Try to make the reward continuous from 0-1.
        return r
```

### 6.3 Designing rewards

Common patterns (return one float per sample):

* **harm-driven:** use scorer-generated scores for defender harm as a key component of the reward
* **Preference methods (DPO/IPO/ORPO):** may not use rewards directly; rely on log-prob differences.
* **Tips:** bound/clip; normalize across a batch; document "higher is worse" vs "higher is better".

### 6.4 implementing `advance(...)`

Default text setting is simple concatenation (below) but you can customize how the next state is created.

```python
def advance(self, context, probe, response):
    return context + probe + response
```


### 6.5 Saving models (HF & non-HF)

* **HF:** `model.save_pretrained(path)` and `tokenizer.save_pretrained(path)`.
* **Non-HF:** `torch.save(model.state_dict(), path)` and a small loader util. Ensure your trainer saves anything else your algorithm needs (e.g., optimizer/scheduler state).

---

## 7. Plug into Sampler / Solver

Your system will be passed to the 'Sampler' and 'Solver'. The 'Trainer' will have access to the system through the sampler (sampler.system).
```python
system = MyHFSystem("gpt2", "gpt2", "gpt2", DetoxifyScorer(), device="cuda")
sampler = ASTSampler(system, PROMPTS, tree_width=2, tree_depth=3)
solver  = DPO(system, beta=0.1)
trainer = Trainer(config=config, sampler=sampler, algorithm=solver)
trainer.train()
```

---

## 8. Debug Checklist

* **Batching:** input list lengths match; outputs align (`[B]` tensors).
* **Gradients:** auditor log-probs require grad; target/baseline under `no_grad`.
* **Masking:** only continuation tokens contribute to `log P(cont | ctx)`.
* **Context window:** `len(ctx_tokens) + max_new_tokens ≤ max_ctx`.
* **Tokenizer differences:** never cross-decode; keep model/tokenizer pairs.
* **Device/type:** tensors on right device/dtype; `pad_token_id` set.
* **Numerics:** watch for `nan/inf`; clip/normalize rewards.
* **Repro:** fixed seeds for rollout sampling and generation settings.
