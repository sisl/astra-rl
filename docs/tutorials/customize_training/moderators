# How to Create a Custom Moderator Class

Moderators play a central role in LM red teaming, acting similarly to reward models in traditional reinforcement learning. Their job is to quantify the reward an adversarial agent receives for reaching a particular state—typically by measuring how harmful or unsafe a target model's output is.

In many RL-based red-teaming setups, the moderator provides the signal that trains the attacker to generate utterances that elicit harmful responses from a target model. This achieves the red-teaming objective by exposing weaknesses in the target model’s safety alignment and highlighting where additional fine-tuning is needed.

To serve this purpose, a moderator must:
- Accept a sequence of target model generations (e.g., text),
- Return a scalar score (e.g., a toxicity value from 0 to 1) indicating the level of harm.

The astra-rl toolbox currently supports text-based moderation using:
- Detoxify, for toxicity classification,
- Llama Guard 3, for a variety of harm categories (e.g., hate speech, threats, etc.).

But the framework is modular—you can define your own moderator class, wrapping any model that takes in your defined StateT and ActionT types (see astra_rl/core/common) and returns a Sequence[float].

This guide walks you through creating a new Moderator subclass.

---

## Step 1: Subclass the Moderator Base Class

To define your own moderator, create a class that inherits from:

```python
Moderator[StateT, ActionT]
```
Where:
- StateT is the type of state your environment uses (e.g., a string prompt)
- ActionT is the type of action your model produces (e.g., a generated response)
For most NLP use cases, both StateT and ActionT are str.

example:
```python
from typing import Sequence
from detoxify import Detoxify
from astra_rl.core.moderator import Moderator

class DetoxifyModerator(Moderator[str, str]):
    def __init__(self, harm_category: str = "toxicity", variant: str = "original"):
        self.model = Detoxify(variant)
        self.harm_category = harm_category
```

---

## Step 2: Implement the moderate Method

You must implement the abstract method:
```python
def moderate(self, x: Sequence[Union[StateT, ActionT]]) -> Sequence[float]:
```

This method:
- Takes a sequence of states and/or actions.
- Returns a sequence of floats, where each float is the moderation score (e.g., toxicity score) for the corresponding input.

example:
```python
def moderate(self, x: Sequence[str]) -> Sequence[float]:
        return self.model.predict(x)[self.harm_category]
```

---

## Step 3: Integrate your moderator

Once your class is defined, you can plug it into the RL pipeline like any other component:

```python
moderator = DetoxifyModerator(harm_category="insult", variant="unbiased")
scores = moderator.moderate(["you are stupid", "have a nice day!"])
```

To train with your custom moderator, modify your problem subclass to instantiate it during initialization:

example:
```python
class ExampleDetoxifyProblem(ASTProblem):
    def __init__(self, device="cpu"):
        # your choice of moderator
        super().__init__(DetoxifyModerator()) ## Plug in your custom moderator here ##

        self.device = device
        self.attacker = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)
        self.target = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
```

---

## Helpful Notes:
- Your moderator can wrap any scoring model—e.g., classifiers, LLMs, rule-based filters—as long as it implements moderate(...) → Sequence[float].

- You can include internal logic to handle tokenization, batching, preprocessing, etc.

- Return one score per input in the same order as received.

- If you're using a library or model that scores multiple types of harm (like Detoxify or llamaguard), your class can expose a harm_category attribute to customize which score to extract.

---

## Full examples:
See the following files for complete, working implementations:
- [astra_rl/moderators/detoxify.py](astra_rl/moderators/detoxify.py) — wraps the Detoxify library
- [astra_rl/moderators/llamaGuard.py](astra_rl/moderators/llamaGuard.py) — wraps Meta’s Llama Guard 3 model