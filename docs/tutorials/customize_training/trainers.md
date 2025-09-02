The Trainer Class is defined at astra_rl/training/trainer.py . It has 3 core responsibilities: 1) it instantiates the optimizer which will update the attacker's policy to lower training loss. 2) it instantiates the training harness which handles collecting online training data (we advise leaving the harness code alone); 3) it conducts the main training loop in the train method. 

To use the Trainer CLass be sure to import the Trainer base class and the training configuration. like so 

```python
from astra_rl import (
    Trainer,
    TrainingConfiguration,
)
```

First, instantiate the training configuration with the training hyperparamaters you desire.
```python 

```
Then, instantiate an instance of the Trainer base class with your training config, environmnet and algorithm.

Look over the simple train() method in the Trainer base class. If you are okay with how this training loop operates (optimizes attacker on training alg loss and DOES NOT 
SAVE THE MODEL), you can simply use the base train method.

If you want a more sophisticated approach that periodically runs the updated model on a development data set and continually saves (in Hugging face format) the best performing model, import the HFASTTrainer class and instantiate it with how frequently you wish to perform evaluations during training, your training config, environment and algorithm.

> Note: The Trainer class will instantiate the training harness. We encourage you to not change the code for the harness and instead adjust training through the trainer class, paramaters, environment or algorithm. 

If you wish to further refine the training loop (including how/when the model is saved during training, adding features such as learning rate scheduling, or implementing a custom training loop design), this tutorial will help you create a custom training class!