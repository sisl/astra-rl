# Moderator

**Moderators** provide the training signal in LM red-teaming. They act much like reward models in RL: given text (typically the target/defender’s reply), they return a scalar score that reflects harm/unsafety. Attackers are then trained—via your chosen solver (e.g., DPO/IPO/PPO)—to produce utterances that elicit high-harm (or otherwise “undesirable”) target responses, revealing weaknesses in the target’s safety alignment.

ASTRA-RL ships with ready-to-use text moderators and a simple interface for writing your own. This guide explains what a moderator does, what’s included, and how to implement/customize your own class.

---

## 1. What Moderators Do

A moderator converts text into a **scalar score** (one score per input). In most setups:

* **Input:** target/defender generations (strings).
* **Output:** `Sequence[float]` scores, e.g., toxicity in `[0, 1]`.

Downstream solvers interpret these scores to train the **attacker**. For preference-based methods (DPO/IPO/ORPO), scores can help form preferences; for policy-gradient methods (PPO/A2C), scores serve directly as rewards/reward components.

---

## 2. Built-in Moderators

ASTRA-RL currently ships with text-based moderators that you can use out of the box:

* **Detoxify** — toxicity classification (and related categories). More info [here](https://github.com/unitaryai/detoxify)
* **Llama Guard 3** — multi-category safety classifier (e.g., hate/threats/harassment). More info [here](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

> These are modular components—swap them freely or use them as templates for your own moderators.

---

## 3. Ways to Customize

### 3.1 Fast path: adapt a built-in

If you only need to change the **category** (e.g., “toxicity” → “insult”), adjust thresholds, or tweak preprocessing/batching, you can wrap or lightly subclass a built-in moderator.

### 3.2 Full control: subclass `Moderator`

For custom scoring models (LLMs, classifiers, rule-based filters), subclass the generic base class and implement one method:

```python
from astra_rl.core.moderator import Moderator
from typing import Sequence, Union, Generic, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")

class MyModerator(Moderator[StateT, ActionT]):
    def moderate(self, x: Sequence[Union[StateT, ActionT]]) -> Sequence[float]:
        ...
```

---

## 4. Required Interface

### 4.1 Type parameters

* `StateT` — your environment’s state type (commonly `str` conversation context).
* `ActionT` — your action type (commonly `str` utterance).

For NLP use cases, both are typically `str`.

### 4.2 `moderate(...)` contract

```python
def moderate(self, x: Sequence[Union[StateT, ActionT]]) -> Sequence[float]:
    """Return one scalar score per input, same order as received."""
```

**Expectations:**

* **Pure function** over the given inputs (no hidden batch size assumptions).
* **Shape:** output length equals input length.
* **Scale/direction:** document whether *higher = more harmful*. (Recommended.)

---

## 5. Best Practices & Sanity Checks

* **Batching:** Vectorize model calls for speed; avoid per-item loops.
* **Preprocessing:** Handle tokenization/normalization inside the class.
* **Calibration:** Keep scores on a consistent scale (e.g., `[0, 1]`) and direction (higher = worse).
* **Throughput vs. latency:** Accumulate inputs into sensible batch sizes.
* **Robustness:** Validate on a small corpus; check extremes and benign inputs.
* **Logging:** Consider returning/recording auxiliary diagnostics (category probabilities, thresholds) for debugging—while still meeting the `Sequence[float]` return type.

---

## 6. How-Tos

### 6.1 Minimal custom moderator (Detoxify wrapper)

```python
from typing import Sequence
from detoxify import Detoxify
from astra_rl.core.moderator import Moderator

class DetoxifyModerator(Moderator[str, str]):
    def __init__(self, harm_category: str = "toxicity", variant: str = "original"):
        self.model = Detoxify(variant)
        self.harm_category = harm_category

    def moderate(self, x: Sequence[str]) -> Sequence[float]:
        # Detoxify returns a dict of category -> scores
        preds = self.model.predict(x)
        return [float(preds[self.harm_category][i]) for i in range(len(x))]
```

### 6.2 Selecting harm categories

If the underlying library/model exposes multiple categories (e.g., Detoxify or Llama Guard 3), surface a `harm_category` (or list of categories) in your constructor. You can:

* return a **single** category’s score, 
* ignore the harm category and return the score for **any violation**, or
* compute a **combined** score (e.g., max/mean across selected categories).

### 6.3 Batching & preprocessing

Inside `moderate(...)`, you’re free to:

* tokenize inputs, truncate/normalize text, strip HTML, etc.;
* split inputs into fixed-size batches to fit device memory;
* run the model on GPU/CPU as configured.

Just be sure to **preserve ordering** and return one scalar per input.

### 6.4 Integrate your moderator into a Problem

Instantiate your moderator in your `Problem` subclass and pass it to the base class:

```python
from transformers import GPT2LMHeadModel, AutoTokenizer
from astra_rl import ASTProblem  # base Problem
from astra_rl.logging import logger

MODEL_NAME = "gpt2"

class ExampleDetoxifyProblem(ASTProblem):
    def __init__(self, device: str = "cpu"):
        # Plug in any custom moderator here
        super().__init__(DetoxifyModerator(harm_category="toxicity"))

        self.device = device
        self.attacker = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)
        self.target   = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
```

> After this, your environment/solver will use the moderator implicitly when computing rewards.

---

## 7. Full Examples

* [astra_rl/moderators/detoxify.py](https://github.com/sisl/astra-rl/blob/main/src/astra_rl/moderators/detoxify.py) — wraps the Detoxify library.
* [astra_rl/moderators/llamaGuard.py](https://github.com/sisl/astra-rl/blob/main/src/astra_rl/moderators/llamaGuard.py) — wraps Meta's Llama Guard 3.

Use these as references when building your own moderator classes.
