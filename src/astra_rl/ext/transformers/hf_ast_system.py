# type: ignore

"""
hf_ast_system.py
Huggingface Transformer extensions to AST system.
Sadly, HF transformers don't play well with typing
so we ignore types here.

https://discuss.huggingface.co/t/static-type-checking-with-mypy-whats-the-official-position/464/4
"""

from typing import Sequence, Iterator, Optional

import torch

from transformers.generation.utils import GenerationMixin
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BatchEncoding,
)
from astra_rl.core.system import ValueFunctionSystem
from astra_rl.core.scorer import Scorer
from astra_rl.methods.ast_system import ASTSystem
from astra_rl.training.trainer import Trainer, TrainingConfiguration
import os
from astra_rl.utils import logger


class HFASTSystem(ASTSystem, ValueFunctionSystem):
    """Huggingface Transformers adaptor for ASTSystem.

    This class extends the ASTSystem to work with Huggingface Transformers models without
    the boilerplate needed to figure out logprobs and rollouts.

    Attributes:
        device (str): The device to run the models on (default is "cpu").
        tester (AutoModelForCausalLM): The tester model used for generating sequences.
        tester_tokenizer (PreTrainedTokenizer): The tokenizer for the tester model.
        target (AutoModelForCausalLM): The target model used for evaluating sequences.
        target_tokenizer (PreTrainedTokenizer): The tokenizer for the target model.
        baseline (AutoModelForCausalLM): The baseline model used for comparison.
        baseline_tokenizer (PreTrainedTokenizer): The tokenizer for the baseline model.

    See astra_rl.methods.ast_system.ASTSystem for more details on usage.
    """

    def __init__(
        self,
        tester_model_id: str,
        target_model_id: str,
        baseline_model_id: str,
        scorer: Scorer[str, str],
        device: str = "cpu",
    ) -> None:
        """Initialize an HFASTSystem instance from Huggingface model IDs.

        Args:
            tester_model_id (str): The model ID for the tester model, must be possible for AutoModelForCausalLM.
            target_model_id (str): The model ID for the target model, must be possible for AutoModelForCausalLM.
            baseline_model_id (Optional[str]): The model ID for the baseline model, if any; otherwise defaults to target model.
            scorer (Scorer): The scorer used to evaluate sequences.
            device (str): The device to run the models on (default is "cpu").
        """
        super().__init__(scorer)

        self.device = device

        # initialize models and tokenizer
        self.tester = AutoModelForCausalLM.from_pretrained(tester_model_id).to(
            self.device
        )
        self.tester_tokenizer = AutoTokenizer.from_pretrained(tester_model_id)

        self.target = AutoModelForCausalLM.from_pretrained(target_model_id).to(
            self.device
        )
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)

        if baseline_model_id is not None:
            self.baseline = AutoModelForCausalLM.from_pretrained(baseline_model_id).to(
                self.device
            )
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_id)
        else:
            # if the defense and baseline are identical, we can use the defense as the baseline
            self.baseline = self.target
            self.baseline_tokenizer = self.target_tokenizer

        # a bunch of models doesn't have padding, so we set the pad token to the eos token
        if self.tester_tokenizer.pad_token_id is None:
            self.tester_tokenizer.pad_token_id = self.tester_tokenizer.eos_token_id
        if self.target_tokenizer.pad_token_id is None:
            self.target_tokenizer.pad_token_id = self.target_tokenizer.eos_token_id
        if self.baseline_tokenizer.pad_token_id is None:
            self.baseline_tokenizer.pad_token_id = self.baseline_tokenizer.eos_token_id

        # set the tokenizer padding and truncation side
        self.tester_tokenizer.padding_side = self.target_tokenizer.padding_side = (
            self.baseline_tokenizer.padding_side
        ) = "left"
        self.tester_tokenizer.truncation_side = (
            self.target_tokenizer.truncation_side
        ) = self.baseline_tokenizer.truncation_side = "left"

    def get_target_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        return self.__get_logprobs(
            self.target, self.target_tokenizer, context, continuation, self.device
        )

    def get_baseline_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        return self.__get_logprobs(
            self.baseline, self.baseline_tokenizer, context, continuation, self.device
        )

    def get_tester_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        return self.__get_logprobs(
            self.tester, self.tester_tokenizer, context, continuation, self.device
        )

    def rollout_prompt_with_tester(self, x: Sequence[str]) -> Sequence[str]:
        return self.__rollout(self.tester, self.tester_tokenizer, x, self.device)

    def rollout_prompt_with_target(self, x: Sequence[str]) -> Sequence[str]:
        return self.__rollout(self.target, self.target_tokenizer, x, self.device)

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return self.tester.parameters()

    @staticmethod
    def __rollout(
        model: GenerationMixin,
        tokenizer: PreTrainedTokenizer,
        prompt: Sequence[str],
        device: str = "cpu",
    ) -> Sequence[str]:
        tokenized_prompt = tokenizer(
            prompt, padding=True, return_tensors="pt", padding_side="left"
        ).to(device)
        output = model.generate(
            **tokenized_prompt,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=32,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=1.0,
        )
        continuation = [
            i[len(j) :]
            for i, j in zip(
                tokenizer.batch_decode(output, skip_special_tokens=True), prompt
            )
        ]
        return continuation

    def value(self, context, continuation):
        # tokenize both context and continuation
        context = self.tokenizer(context)
        continuation = self.tokenizer(continuation)

        # create a mask such that the context is masked out
        # in order to only compute logprobs of P(continuation|context)
        combined_mask = [
            [False] * len(i) + [True] * len(j)
            for i, j in zip(context.input_ids, continuation.input_ids)
        ]

        # combine context + continuation; compute how much to pad
        combined = [i + j for i, j in zip(context.input_ids, continuation.input_ids)]
        max_length = max(len(i) for i in combined)

        # pad the combined input and context mask
        # use eos_token as padding
        combined = [
            i + [self.tokenizer.eos_token_id] * (max_length - len(i)) for i in combined
        ]
        combined_mask = [i + [False] * (max_length - len(i)) for i in combined_mask]
        attention_mask = [
            [True] * len(i) + [False] * (max_length - len(i)) for i in combined_mask
        ]

        # move things to torch and cuda
        combined = torch.tensor(combined).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        combined_mask = torch.tensor(combined_mask).to(self.device)

        # run inference
        output = self.tester(
            input_ids=combined, attention_mask=attention_mask, output_hidden_states=True
        )
        projected = self.vf(output.hidden_states[-1])

        # compute per-token likelihoods
        gathered = projected.masked_fill(
            ~(combined_mask.unsqueeze(-1).repeat(1, 1, projected.size(-1))), 0.0
        )

        return gathered[:, :-1]

    @staticmethod
    def __get_logprobs(
        model: GenerationMixin,
        tokenizer: PreTrainedTokenizer,
        context_text: Sequence[str],
        continuation_text: Sequence[str],
        device: str = "cpu",
    ) -> torch.Tensor:
        # tokenize both context and continuation
        context: BatchEncoding = tokenizer(context_text)
        continuation: BatchEncoding = tokenizer(continuation_text)

        # create a mask such that the context is masked out
        # in order to only compute logprobs of P(continuation|context)
        combined_mask = [
            [False] * len(i) + [True] * len(j)
            for i, j in zip(context.input_ids, continuation.input_ids)
        ]

        # combine context + continuation; compute how much to pad
        combined = [i + j for i, j in zip(context.input_ids, continuation.input_ids)]
        max_length = max(len(i) for i in combined)

        # pad the combined input and context mask
        # use eos_token as padding
        combined = [
            i + [tokenizer.eos_token_id] * (max_length - len(i)) for i in combined
        ]
        combined_mask = [i + [False] * (max_length - len(i)) for i in combined_mask]
        attention_mask = [
            [True] * len(i) + [False] * (max_length - len(i)) for i in combined_mask
        ]

        # move things to torch and cuda
        combined = torch.tensor(combined).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        combined_mask = torch.tensor(combined_mask).to(device)

        # run inference
        logits = (
            model(input_ids=combined, attention_mask=attention_mask)
            .logits[:, :-1]
            .log_softmax(dim=-1)
        )

        # compute likelihoods per token
        gathered = logits.gather(-1, combined[:, 1:].unsqueeze(-1)).squeeze(-1)
        # mask out padding tokens' probabilities (i.e. set them to e^0 = 1).
        gathered = gathered.masked_fill(~combined_mask[:, 1:], 0.0)

        # Return per-token logprobs instead of aggregating
        return gathered


class HFASTConfiguration(TrainingConfiguration):
    def __init__(self):
        super().__init__(
            lr=1e-5,
            batch_size=4,
            optimizer="adamw",
            gradient_accumulation_steps=1,
            training_steps=1000,
            num_episodes_per_experience=2,
        )


class HFASTTrainer(Trainer):
    """
    Subclass that reuses base init (harness, optimizer, counters) and adds:
      - periodic evaluation on a dev set
      - checkpointing of the tester/tokenizer
    """

    def __init__(
        self,
        config,
        environment,
        algorithm,
        *,
        dev_prompts=None,
        eval_every=200,
        ckpt_dir="checkpoints",
    ):
        super().__init__(config, environment, algorithm)
        self.dev_prompts = dev_prompts or []
        self.eval_every = max(1, int(eval_every))
        self.best_score = float("-inf")
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.sampler = self.harness.sampler
        self.system = self.sampler.system

    # helper function that saves the tester model in HF format
    def save(self, step: int | None, tag: str = "step"):
        if tag == "best":
            out = os.path.join(self.ckpt_dir, "best")  # single fixed path
        else:
            out = os.path.join(self.ckpt_dir, f"{tag}-{step}")

        # Save tester/target in HF format
        os.makedirs(out, exist_ok=True)
        self.system.tester.save_pretrained(out)
        self.system.tokenizer.save_pretrained(out)
        logger.info(f"Saved checkpoint to {out}")

    @torch.no_grad()
    def eval_epoch(self, step: int, tag: str = "dev"):
        if not self.dev_prompts:
            return float("nan")

        logger.info(f"EVALUATING after training step {step}...")
        rewards = []
        num_dev_prompts = len(self.dev_prompts)

        # TODO batch later for speed
        for indx, i in enumerate(self.dev_prompts):
            if indx % 30 == 0:
                logger.info(f"EVAULATED {indx}/{num_dev_prompts} steps...")
            # perform a sigle eval rollout per dev prompt and collect a list of rewards
            eval_rollout = self.sampler.eval_rollout(i)
            final_rollout_reward = self.sampler.final_reward(eval_rollout)
            rewards += [final_rollout_reward]

        logger.info(f"EVAULATED {indx}/{num_dev_prompts} steps...")
        dev_score = sum(rewards) / len(rewards)

        if dev_score > self.best_score:
            logger.info(f"NEW BEST! {round(dev_score, 3)}")
            logger.info({"training/dev_score": dev_score}, step=step)
            self.save(step, tag="best")
        else:
            logger.info(
                f"Dev score did not improve: {round(dev_score, 3)} vs {round(self.best_score, 3)}"
            )

    # over-write the base Train class's train method to include eval and save
    def train(self):
        for step_num in range(self.config.training_steps):
            # collect some experiences using current weights
            buf = self.harness.experience()  # <- this is a torch dataloader
            for i in buf:
                # we compute the loss using the algorithm we chose
                loss, step_logs = self.harness.step(i)

                # this is normal optimization; feel free to do weight decay, etc.
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Add custom and algorithm external logging here (e.g., step number)
                step_logs["step"] = step_num
                self.harness.log_current_step(step_logs)

                # every x number of training steps, run a dev set eval and save the best model so far
                if (step_num + 1) % self.eval_every == 0:
                    self.eval_epoch(step=step_num + 1, tag="dev")


class HFEvaluationSystem(HFASTSystem):
    """
    Minimal evaluation wrapper for non-GPT2 HF models.

    - Initializes the HFASTSystem using a sensible base id so HFASTSystem
      sets up tokenizers/target/baseline exactly as it does for training.
    - Replaces the tester model weights from `tester_checkpoint`.
    - If the checkpoint contains a tokenizer, prefers that tokenizer; otherwise
      leaves the tokenizers configured by HFASTSystem intact.

    Note: This class intentionally does NOT apply GPT-2 specific fixes
    (pad_token/eos_token mapping or max_ctx handling). Use a GPT2-specific
    subclass for GPT-2 quirks.
    """

    def __init__(
        self,
        tester_checkpoint: str,
        tester_base_model_id: Optional[str],
        target_model_id: str,
        device: str = "cpu",
        scorer: Optional[Scorer] = None,
        baseline_model_id: Optional[str] = None,
    ) -> None:
        # Choose a safe model id to give the parent: prefer an explicit base id if provided,
        # otherwise pass the checkpoint itself. HFASTSystem will create tokenizers from that id.
        base_id_for_super = tester_base_model_id or tester_checkpoint

        # Initialize HFASTSystem exactly as it would for training (so tokenizers are set up the same way)
        super().__init__(
            tester_model_id=base_id_for_super,
            target_model_id=target_model_id,
            baseline_model_id=baseline_model_id,
            scorer=scorer,
            device=device,
        )

        self.device = device

        # Replace tester weights with the trained checkpoint (local HF dir or hub id).
        try:
            self.tester = AutoModelForCausalLM.from_pretrained(tester_checkpoint).to(
                self.device
            )
            logger.info(f"Loaded tester model weights from: {tester_checkpoint}")
        except Exception as e:
            # fallback: keep tester loaded by super().__init__ (from base_id_for_super)
            logger.warning(
                f"Failed to load tester checkpoint '{tester_checkpoint}': {e}. "
                "Using tester model loaded by HFASTSystem (from base id)."
            )

        # If the checkpoint contains tokenizer files, prefer them. Otherwise keep the tokenizers that HFASTSystem created.
        try:
            # This will succeed when the checkpoint folder contains tokenizer files (tokenizer.json, vocab.json, merges.txt, etc.)
            ckpt_tokenizer = AutoTokenizer.from_pretrained(tester_checkpoint)
            self.tester_tokenizer = ckpt_tokenizer
            logger.info(f"Loaded tester tokenizer from checkpoint: {tester_checkpoint}")
        except Exception:
            # No tokenizer in checkpoint — leave the tokenizer created by HFASTSystem (from base_id_for_super)
            logger.debug(
                "No tokenizer found in tester_checkpoint; using tokenizer from HFASTSystem initialization."
            )

        # canonical alias expected by other code (trainer.save uses system.tokenizer)
        self.tokenizer = self.tester_tokenizer

        logger.info("HFEvaluationSystem initialized (non-GPT2 path).")


__all__ = ("HFASTSystem",)
