Do you want to train a Huggingface adversary using an ASTRA-supported algorithm (DPO, IPO, PPO*) and problem formulation (ASTPrompter, Perez*, MALIBU*, Hong*)? Then this quick start guide will walk you through every step to training a red-teaming adversary. 
TARGET, BASELINE, ATTACKER

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

2) Create a python file for your training code (ie. mytrain.py)

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

The supported framework (ASTPrompter) red-teams for harmful behaviors in conversational LMs and therefore uses the ConvoKit Reddit corpus (small) to serve as the initial prompts. To use this data, make sure you have the prompts_reddit_train.json file in your local directory and read the prompts in as a comma seperated list as shown below.

```python 
with open("prompts_reddit_train.json") as f:
        PROMPTS = json.load(f)
```

If you would like to use a different dataset to initialize training rollouts, simply load your json file instead! Just make sure that your json file is formatted as a list of comma seperated strings.

5) Set your device:
If you have acsess to a GPU:
```python
DEVICE = "cuda" 
```
This will allow you train larger adversaries and use larger moderator models. Since we want to make this framework available to people with a variety of computational capabilities, we put the occonus of properly allocating recources to you, the user. 


If you do not have acsess to a GPU, you can still use this toolbox to train smaller adversaries for red teaming! Try finetunning smaller adversarial models such as GPT2 and using lighter-weight moderators (detoxify)!
```python
DEVICE = "cpu" 
```

6) Instatiate your problem (choose adversary, defender, reference, and moderator models!)
To facilitate your experience, we have created a problem wrapper that supports any hugging face model!

Adversary/Defender/Reference Hugging face models:

Simply instantiate your problem with the HFASTProblem class and provide the name of the hugging face models you would like to use. Additionally, you will need to provide a moderator that will evalute adversary/defender utterances to determine rewards. We have two moderator classes pre-configured and ready for use: DetoxifyModerator and LlamaGuardModerator()*.

```python
# example for instatiating a problem with gpt2 as the adversary, defender, andreference model with a detoxify moderator! (cpu-friendly)
problem = HFASTProblem("gpt2", "gpt2", "gpt2", DetoxifyModerator(), DEVICE)

# example for instatiating a problem with gpt2 as the adversary, llama3 as the defender, and reference model with a llamaguard moderator!(gpu necessary)
problem = HFASTProblem("gpt2", "llama??", "gpt2", LlamaGuardModerator(), DEVICE)

``` 

Want to use a non-hugging face attacker, target, or baseline? No Problem!
Go to: astra-rl/docs/tutorials/custom_models.md
to create a problem subclass that fits your needs!  

Want to use a moderator besides llamaguard or detoxify? It's easy!
Go to: astra-rl/docs/tutorials/custom_moderators.md
to learn how to create a class for your moderator and integrate it into your problem class!

7) 