# How to Customize the Environment

The environment defines how attacker–target interactions are structured and collected for training. It controls **how rollouts are generated**, **how trajectories are organized**, and **what data is passed to the solver**.

In ASTRA-RL, the default environment is `ASTEnvironment`, which constructs a tree of attacker–target interactions as used in the [ASTPrompter](https://arxiv.org/abs/2407.09447) paper. You can fully customize this structure to support:
- Multi-agent interactions
- Turn-based or fixed-length rollouts
- Flat trajectories instead of trees
- New state advancement logic
- Different reward shaping strategies

---

## Step 1: Subclass the Base Environment

All custom environments must subclass:

```python
from astra_rl.core.environment import Environment

class MyCustomEnvironment(Environment[StateT, ActionT]):
    ...
```     
This base class defines the interface between the environment and the training harness.

---
## Step 2: Implement the rollout method

Your custom class must implement:

```python
def rollout(self, seed: Optional[int] = None) -> Graph[StateT, ActionT]
```

The rollout() method should return a Graph object, which holds a tree or list of conversation trajectories, each node containing at a minimum:
- state (e.g. prompt so far)
- action (e.g. attacker utterance)
- target_response (optional)
- reward (float)

Within your rollout() method, you can utilize the instance of ASTProblem your encironment will interact with, providing...
- rollout_prompt_with_attacker(prompts: Sequence[str]) → Sequence[str]
- rollout_prompt_with_target(prompts: Sequence[str]) → Sequence[str]
- reward(prompts, attacks, defenses) → Sequence[float]
- advance(prompt, attack, defense) → str: updates the conversation state

You can call these methods directly inside your environment to implement the rollout logic of your choice.

## Tips and best practices
- Use Node(...) to represent each step of a rollout. Each node includes:

```python
Node(state: str, action: str, target_response: str, reward: float, children: List[Node])
```

- Return a Graph(prompt, nodes) from your rollout() function.
- advance() helps build the new state after attacker and target interact.
- Want to add additional metadata? Subclass Node and modify Graph accordingly.