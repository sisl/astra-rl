Do you want to train a Huggingface attacker using an ASTRA-supported algorithm (DPO, IPO, PPO*) and problem formulation (ASTPrompter, Perez*, MALIBU*, Hong*)? Then this quick start guide will walk you through every step to training a red-teaming attacker. 
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
To train your attacker, you will need to provide a large dataset of comma-seperated strings that will serve as the prompts that initiate attacker-target conversation rollouts during training. These rollouts (scored by the moderator) will serve as the training data for the attacker, teaching the attacker what actions it should take to ellicit harmful responses from the target. Therefore, it is important that the initial prompts you provide are relavent to the red-teaming scenario you care about. For example, if you want to red-team medical diagnosis LMs, your initial prompts should reflect the prompts a medical diagnosis LM would likely see during operation: "I have a cut that is swollen and red. What do you think is wrong?" ect.

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

6) Instatiate your problem (choose attacker, target, baseline, and moderator models!)
To facilitate your experience, we have created a problem wrapper that supports any hugging face model! Simply instantiate your problem with the HFASTProblem class and provide the name of the hugging face models you would like to use. Additionally, you will need to provide a moderator that will evalute adversary/defender utterances to determine rewards. We have two moderator classes pre-configured and ready for use: DetoxifyModerator and LlamaGuardModerator()*.

```python
# example for instatiating a problem with gpt2 as the adversary, defender, andreference model with a detoxify moderator! (cpu-friendly)
problem = HFASTProblem("gpt2", "gpt2", "gpt2", DetoxifyModerator(), DEVICE)

# example for instatiating a problem with gpt2 as the adversary, llama3 as the defender, and reference model with a llamaguard moderator!(gpu necessary)
problem = HFASTProblem("gpt2", "llama??", "gpt2", LlamaGuardModerator(), DEVICE)

``` 

Want to use a non-hugging face attacker, target, or baseline? Want to rollout the models in a specific way? No Problem!
Go to: astra-rl/docs/tutorials/customize/problems.md
to create a problem subclass that fits your needs!  

Want to use a moderator besides llamaguard or detoxify? It's easy!
Go to: astra-rl/docs/tutorials/customize/moderators.md
to learn how to create a class for your moderator and integrate it into your problem class!

7) Instantiate your environment class
- the environment class determines how training rollouts are performed and collected. For example, the pre-configured ASTEnvironemnt class generates two actions per conversation state, generating an experience graph that is horizon (H) deep and 2*H wide. This graph supports contrastive preference learning algorithms such as DPO and IPO. However, you may wish to train with an algorithm that does not require labeled action pairs and only want to generate a single, linear conversation rollout per prompt. 

example
```python
env = ASTEnvironment(problem, PROMPTS)
```

Do the current environment classes not suite your red teaming needs? Go to:
astra-rl/docs/tutorials/custom_environments
To create a custom environment subclass that rolls out attacker-target conversation trajectories in a way that fits your needs! Your custom environement can
        - control conversation horizon
        - control conversation branching factor
        - control what information is saved at each step/node in the conversation

8) Chose your solver (algorithm) and optimizer
supported algorithms: DPO, PPO

```python
solver = DPO(problem)
optimizer = AdamW(problem.parameters(), lr=1e-5)
```

Interested in integrating a uniqe or not-yet-implemented learning algorithm? Please visit 
astra-rl/docs/tutorials/customize/solvers
to see how you can smoothly integrate any learning algorithm into the ASTRA_RL training pipeline!

9) Create your training harness
this controls training paramaters such as number of episodes per exerience and batch size. 
```python
harness = Harness(
        env,
        solver,
        num_episodes_per_experience=2,
        use_wandb=True,
        dataloader_kwargs={"batch_size": 4},
    )
```

For more information on the training harness, visit:
astra-rl/docs/tutorials/customize/harness

10) Start training!
Perform a training loop through your designed number of training steps. Your buffer (buf) will hold onto 
your rollouts and perform steps that finetune attacker model weights using your chosen learning algorithm. The example below shows normal optimization, but our flexible framework allows you to truly customize the learning experience!

```python
# optimization step
    for step in range(1000):
        # collect some experiences using current weights
        buf = harness.experience()  # <- this is a torch dataloader
        for i in buf:
            # we compute the loss using the algorithm we chose
            loss, step_logs = harness.step(i)
            # this is normal optimization; feel free to do weight decay, etc.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Add custom and algorithm external logging here (e.g., step number)
            # TODO: Do we want multiple logs values per step (iterated over experience buffer)?
            # TODO: Do we want to add other things here to logging?
            step_logs["step"] = step
            harness.log_current_step(step_logs)
```

A fully-implemented, hugging-face ASTRA_RL training pipeline is implemented at 
astra-rl/examples/ast_hf.py
