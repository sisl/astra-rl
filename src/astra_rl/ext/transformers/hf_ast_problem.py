# type: ignore

"""
hf_ast_problem.py
Huggingface Transformer extensions to AST problem.
Sadly, HF transformers don't play well with typing
so we ignore types here.

https://discuss.huggingface.co/t/static-type-checking-with-mypy-whats-the-official-position/464/4
"""

from typing import Sequence, Iterator

import torch

from transformers.generation.utils import GenerationMixin
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BatchEncoding,
)

from astra_rl.core.moderator import Moderator
from astra_rl.methods.ast import ASTProblem


class HFASTProblem(ASTProblem):
    def __init__(
        self,
        attacker_model_id: str,
        target_model_id: str,
        baseline_model_id: str,
        moderator: Moderator[str, str],
        device: str = "cpu",
    ) -> None:
        super().__init__(moderator)

        self.device = device

        # initialize models and tokenizer
        self.attacker = AutoModelForCausalLM.from_pretrained(attacker_model_id).to(
            self.device
        )
        self.attacker_tokenizer = AutoTokenizer.from_pretrained(attacker_model_id)

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
        if self.attacker_tokenizer.pad_token_id is None:
            self.attacker_tokenizer.pad_token_id = self.attacker_tokenizer.eos_token_id
        if self.target_tokenizer.pad_token_id is None:
            self.target_tokenizer.pad_token_id = self.target_tokenizer.eos_token_id
        if self.baseline_tokenizer.pad_token_id is None:
            self.baseline_tokenizer.pad_token_id = self.baseline_tokenizer.eos_token_id

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

    def get_attacker_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        return self.__get_logprobs(
            self.attacker, self.attacker_tokenizer, context, continuation, self.device
        )

    def rollout_prompt_with_attacker(self, x: Sequence[str]) -> Sequence[str]:
        return self.__rollout(self.attacker, self.attacker_tokenizer, x, self.device)

    def rollout_prompt_with_target(self, x: Sequence[str]) -> Sequence[str]:
        return self.__rollout(self.target, self.target_tokenizer, x, self.device)

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return self.attacker.parameters()

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

        # compute likelihoods
        gathered = logits.gather(-1, combined[:, 1:].unsqueeze(-1)).squeeze(-1)
        gathered = gathered.masked_fill(~combined_mask[:, 1:], 0.0)
        logprobs = gathered.sum(dim=-1)

        return logprobs


__all__ = ("HFASTProblem",)
