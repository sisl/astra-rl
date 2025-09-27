When performing an evaluation, you must first create a problem that properly calls your trained attacker model.

If you are using a HuggingFace model that does not have fixed length (eg not GPT2) such as llama3, you are in luck! You can simply use the HFASTProblem and pass in the path to your attacker (to the checkpoint file created during training) as the attacker parameter during problem instantiation.
> from astra_rl.ext.transformers import HFASTProblem

If you are using a GPT2-based attacker model, you are also in luck. We have already created a problem class that will take in the path to your attacker model checkpoint and instanitate your problem correctly. 
> from ast_eval_test.py import EvaluationProblem

However, if you are using a completely custom pre-trained attacker or target, you will have to create a custom problem class that instantiates your models/tokenizers and performs a rollout step (text generation given context) for each model. Please see customize_training/profblems.md to learn more about how to create a custom problem class. 

> Note: If you already created a custom problem class for training, it will likely be best to subclass from that problem and simply change the instantiation to read in your trained attacker. To see an example of this, please see the creation of the EvaluationProblem shown in ast_eval_test.py!