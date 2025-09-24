# How to Customize the Problem (HF or non-HF)

**Problems** encapsulate *models + tokenization + rollout + log-probabilities + rewards*. Environments call your Problem to:

* sample attacker/target continuations,
* compute log-probs for learning objectives (e.g., DPO/IPO/PPO),
* advance the state,
* acsess attacker parameters,
* compute rewards (scalar or per-step).

Most users can subclass `ASTProblem` (text) or use the HF convenience `HFASTProblem`. If you’re integrating a custom or non-HF model, implement the same small API.

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
* **`HFASTProblem`** — Hugging Face adaptor that subclasses ASTProblem and adds tokenization, generation, and log-prob computation for any HF model that does not have a fixed length (eg. not GPT2, but works for other models like llama models).

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

Every Problem must implement the following **batched** methods (lists in, tensors/lists out, index-aligned):

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

To create your custom Problem class, we encourage you to find what existing problem is the closest fit to your desired Problem and subclass from there. 

For example, if you are still using huggingface models but want to change how the state advances or how the reward is calculated, you should subclass from the HFASTProblem, define the methods you wish to change, and let the pre-defined methods in HFASTProblem remain. 

The following code shows how to subclass from the base Problem class and where you should implement your custom methods to create the changes you desire. 

```python
class MyProblem(Problem[str, str]):
    def __init__(self, moderator, attacker_model, target_model, baseline_model, device="cuda"):
        super().__init__(moderator)
        self.device = device

        # set your attacker, target, and baseline models
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
        # Implement your own encode/combine/mask logic 
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
        # calculate your custom reward here! 
        # return a scalar value. Note that binary signals are not as helpful for training. Try to make the reward continuous from 0-1. 
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

