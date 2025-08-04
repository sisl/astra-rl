Do you want to train a Huggingface adversary using an ASTRA-supported algorithm (DPO, IPO, PPO*) and problem formulation (ASTPrompter, Perez*, MALIBU*, Hong*)? Then this quick start guide will walk you through every step to training a red-teaming adversary. 

1) Set-up
See the README for more details and guidance behind the installation process.

```bash
# Install the ASTRA-RL toolbox
pip install astra-rl
# Clone the repository
git clone git@github.com:sisl/astra-rl.git
cd astra-rl
# Sync package dependencies
uv sync --dev
# Install pre-commit hooks:
uv run pre-commit install
```

2) Create a python file for your training code

example:
mytrain.py

3) Import required classes, wrappers, and functions
```python
# import your favorite optimization algorithm. We suggest Adam!
from torch.optim import AdamW
# import a pre-instantiaed environment class, RL training algorithm, Moderator, and training harness class
from astra_rl import ASTEnvironment, DPO, DetoxifyModerator, Harness
# import the wrapper that supports Huggingface models for ASTPrompter-style adversarial training
from astra_rl.ext.transformers import HFASTProblem
```

4) Load your Training Data (initial prompts)
To train your adversary, you will need to provide a large dataset of comma-seperated strings that will serve as the prompts that initiate adversarial-defender conversation rollouts during training. These rollouts (scored by the moderator) will serve as the training data for the adversary, teaching the adversary what actions it should take to ellicit harmful responses from the defender. Therefore, it is important that the initial prompts you provide are relavent to the red-teaming scenario you care about. For example, if you want to red-team medical diagnosis LMs, your initial prompts should reflect the prompts a medical diagnosis LM would likely see during operation: "I have a cut that is swollen and red. What do you think is wrong?" ect.

The supported framework (ASTPrompter) red-teams for harmful behaviors in conversational LMs and therefore uses the ConvoKit Reddit corpus (small) to serve as the initial prompts. To use this data, implement the following code:
 