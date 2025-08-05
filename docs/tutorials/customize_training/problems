# How to Create a Custom Problem Class for Red-Teaming with ASTRA-RL

The `Problem` class defines the core logic of how attacker-target interactions occur in reinforcement-learning-based red-teaming. It specifies:

- How the attacker and target interact and advance the conversation state.
- How reward signals are computed from attacker-target interactions.
- How a model rollout step is performed.
- What models and tokenizers are used for attacker, target, and baseline (if applicable).

By subclassing the `Problem` or `ASTProblem` class, you can customize any of these aspects to fit your particular red-teaming scenario, algorithm, or dataset. In general, we reccomend subclassing from the ASTProblem or HFASTProblem whenever you can and then over-writing methods or definitions to suite your needs. 

This guide will show you how to:

- Create your own custom subclass.
- Integrate your own models and tokenizers.
- Customize reward computation and rollout logic.

---

## Understanding the Base Class

The base `ASTProblem` class provides default implementations suitable for the ASTPrompter approach:

- **`advance`**: Defines how a prompt advances given an attacker action and a target response.
- **`reward`**: Defines how rewards are calculated from attacker-target interactions, typically using toxicity scoring and perplexity measures.
- **`rollout_prompt_with_attacker` / `rollout_prompt_with_target`**: Methods to generate attacker and target model outputs from a given context.
- **`parameters`**: Specifies model parameters for optimization.

You should subclass this base class to preserve and extend this default behavior. If you want to change a method, simply define it in your subclass and it will over-right the original implementation while preserving the rest of the base class functionality.

---

## How to create a subclass with custom models/tokenizers

To subclass your own `Problem`, follow this template:

```python
from astra_rl.methods.ast_problem import ASTProblem
from astra_rl.core.moderator import Moderator

class MyCustomProblem(ASTProblem):
    def __init__(self, moderator: Moderator[str, str], my_custom_param: float = 1.0):
        super().__init__(moderator)
        self.my_custom_param = my_custom_param

    def advance(self, state: str, action: str, response: str) -> str:
        # Example: Simply concatenate with separators
        return f"{state}\n[Attacker]: {action}\n[Target]: {response}"

    def reward(self, contexts, attacks, responses):
        # Implement your own reward logic here
        scores = self.moderator.moderate(responses)
        return [score * self.my_custom_param for score in scores]

    def rollout_prompt_with_attacker(self, prompts):
        # Implement custom attacker rollout logic
        raise NotImplementedError

    def rollout_prompt_with_target(self, prompts):
        # Implement custom target rollout logic
        raise NotImplementedError

    def parameters(self):
        # Implement if your problem has trainable model parameters
        return []
```

---
## Step 3: Integrating Your Own Models (Attacker, Target, or Baseline)

If you are using huggingface models, save time by subclassing from our HFASTProblem base class which takes in any huggingface model names for the attacker, target, and baseline. Additionally, you can integrate any pretrained language model by loading the model and tokenizer in your constructor. Here's how to do it clearly and correctly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Create your subclass from the base Problem (most bare-boned) or the base ASTPrompter class (takes care of rollout and log probability methods)
class MyHuggingFaceProblem(ASTProblem):
    def __init__(self, attacker_model_id: str, target_model_id: str, moderator, device="cuda"):
        super().__init__(moderator)
        self.device = device

        # Load your models and tokenizers
        self.attacker = AutoModelForCausalLM.from_pretrained(attacker_model_id).to(self.device)
        self.attacker_tokenizer = AutoTokenizer.from_pretrained(attacker_model_id)

        self.target = AutoModelForCausalLM.from_pretrained(target_model_id).to(self.device)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)

    # required method to perform one step of a rollout in a batched manner: must take in a state (eg. conversation so far) and return a list of continuations (attacker utterances) in the corresponding order
    def rollout_prompt_with_attacker(self, prompts):
        inputs = self.attacker_tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.attacker.generate(**inputs, max_new_tokens=32)
        generated_texts = self.attacker_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        continuations = [gen[len(prompt):] for prompt, gen in zip(prompts, generated_texts)]
        return continuations

    def rollout_prompt_with_target(self, prompts):
        inputs = self.target_tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.target.generate(**inputs, max_new_tokens=32)
        generated_texts = self.target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        continuations = [gen[len(prompt):] for prompt, gen in zip(prompts, generated_texts)]
        return continuations

    def parameters(self):
        return self.attacker.parameters()
```

---

## Step 4: Customizing Rollout Logic
To customize how attacker-target rollouts are performed, override these methods clearly:

rollout_prompt_with_attacker(prompts: Sequence[str]) → Sequence[str]

rollout_prompt_with_target(prompts: Sequence[str]) → Sequence[str]

For example, you might use sampling strategies, temperature adjustments, or custom stopping criteria:
```python
def rollout_prompt_with_attacker(self, prompts):
    inputs = self.attacker_tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
    outputs = self.attacker.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )
    texts = self.attacker_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [text[len(prompt):] for prompt, text in zip(prompts, texts)]
```

---
## Step 5: Customizing Reward Logic
Override the reward method to compute your custom reward signal. Typically, you'll combine toxicity, relevance, perplexity, or other metrics:
```python
def reward(self, contexts, attacks, responses):
    attack_scores = self.moderator.moderate(attacks)
    response_scores = self.moderator.moderate(responses)
    combined = [(a_score + r_score) / 2.0 for a_score, r_score in zip(attack_scores, response_scores)]
    return combined

```

## Tips and Best Practices:
Always clearly document parameters and logic for your custom problem class.

Ensure models and tokenizers are device-aware (e.g., GPU-compatible).

Thoroughly test your rollouts independently before integrating them into the full RL loop.

For debugging, add verbose logging to track input-output sequences.

## Further Reading and Examples
Default ASTPrompter implementation: ASTProblem

HuggingFace-compatible subclass example: HFASTProblem

Environment customization guide: Custom Environments