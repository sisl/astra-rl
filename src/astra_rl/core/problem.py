"""
problem.py
Generic class of an AstraProblem
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence, Dict, Generic, Union, Iterator

import torch

from astra_rl.logging import logger
from astra_rl.core.moderator import Moderator
from astra_rl.core.common import StateT, ActionT


class Problem(ABC, Generic[StateT, ActionT]):
    """
    A problem is defined by
    - a dataset
    - models (defense, adversary, baseline) + how to call them
    - reward formulation
    """

    def __init__(self, moderator: Moderator[StateT, ActionT]) -> None:
        # we check all asserts once, and then disable them
        self._disable_asserts: Dict[str, bool] = defaultdict(bool)
        self.moderator = moderator

    @abstractmethod
    def get_target_logprobs(
        self, context: Sequence[StateT], continuation: Sequence[ActionT]
    ) -> torch.Tensor:
        """Evaluates P(continuation|context) on *model under test*.

        Args:
            context (Sequence[str]): Sequence of strings, where each string is a context on which the
                                 continuation's probability is conditioned.
            continuation (Sequence[str]): Sequence of strings, where each string is a continuation whose
                                      probability is measured.

        Note:
            This should be batched; i.e., len(context) == len(continuation) and each
            represents a batch element.

        Returns:
            torch.Tensor: The log probabilities of the continuations given their contexts.
        """

        pass

    @abstractmethod
    def get_baseline_logprobs(
        self, context: Sequence[StateT], continuation: Sequence[ActionT]
    ) -> torch.Tensor:
        """Evaluates P(continuation|context) on *attacker's baseline distribution* for KL
           divergence measurements.

        Args:
            context (Sequence[str]): Sequence of strings, where each string is a context on which the
                                 continuation's probability is conditioned.
            continuation (Sequence[str]): Sequence of strings, where each string is a continuation whose
                                      probability is measured.

        Note:
            This should be batched; i.e., len(context) == len(continuation) and each
            represents a batch element. Note that this is *not* the defender's model, but
            rather the baseline model used for measuring KL divergence to make sure that
            the trained attacker stays an LM.

        Returns:
            torch.Tensor: The log probabilities of the continuations given their contexts.
        """

        pass

    @abstractmethod
    def get_attacker_logprobs(
        self, context: Sequence[StateT], continuation: Sequence[ActionT]
    ) -> torch.Tensor:
        """Evaluates P(continuation|context) on *attacker*. This must return tensor w/ grads!

        Args:
            context (Sequence[str]): Sequence of strings, where each string is a context on which the
                                 continuation's probability is conditioned.
            continuation (Sequence[str]): Sequence of strings, where each string is a continuation whose
                                      probability is measured.

        Note:
            This should be batched; i.e., len(context) == len(continuation) and each
            represents a batch element.

        Returns:
            torch.Tensor: The log probabilities of the continuations given their contexts.
        """

        pass

    @abstractmethod
    def rollout_prompt_with_attacker(self, x: Sequence[StateT]) -> Sequence[ActionT]:
        """Rolls out the prompt with the attacker model. Do *not* return the prompt.

        a ~ \\pi(s)

        Args:
            x (Sequence[str]): Sequence of strings representing the prompt to be rolled out.

        Returns:
            Sequence[str]: The rolled out prompt with the adversary model.
        """
        pass

    @abstractmethod
    def rollout_prompt_with_target(self, x: Sequence[StateT]) -> Sequence[StateT]:
        """Rolls out the prompt with the model under test. Do *not* return the prompt.

        s' ~ \\sum_a T(s, a)

        Args:
            x (Sequence[str]): Sequence of strings representing the prompt to be rolled out.

        Returns:
            Sequence[str]: The rolled out prompt with the adversary model.
        """
        pass

    @abstractmethod
    def advance(self, context: StateT, attack: ActionT, response: StateT) -> StateT:
        """Given a context and continuation, returns the next state.

        Args:
            context (str): Sequence of strings representing the context.
            attack (str): Sequence of strings representing the attack given context.
            response (str): Sequence of strings representing the defense against attack.

        Returns:
                str: The next state after applying the continuation to the context.
        """
        pass

    @abstractmethod
    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        """Return the trainable parameters in this problem.

        Returns:
            Iterator[torch.nn.parameter.Parameter]: An iterator over the trainable parameters.
            usually just by calling model.parameters()
        """
        pass

    @abstractmethod
    def reward(
        self,
        context: Sequence[StateT],
        attack: Sequence[ActionT],
        response: Sequence[StateT],
    ) -> Sequence[float]:
        pass

    ##### Utility methods for validation and checks #####

    def _check_continuation(
        self,
        check_key: str,
        context: Sequence[StateT],
        continuation: Sequence[Union[ActionT, StateT]],
    ) -> None:
        if self._disable_asserts[check_key]:
            return
        self._disable_asserts[check_key] = True

    def _check_logprobs(
        self,
        check_key: str,
        logprobs: torch.Tensor,
        ctx_length: int,
        requires_grad: bool = False,
    ) -> None:
        if self._disable_asserts[check_key]:
            return
        # check that logprobs is a tensor and has gradients
        assert isinstance(logprobs, torch.Tensor), (
            "Attacker logprobs must be a torch.Tensor."
        )
        if requires_grad:
            assert logprobs.requires_grad, (
                "Attacker logprobs must carry gradient information."
            )
        # check that the size of the tensor is B x 1, where B is the batch size
        if logprobs.dim() == 1:
            logprobs = logprobs.unsqueeze(1)
        assert logprobs.dim() == 2, (
            "Attacker logprobs must be a 2D tensor (B, 1) or a 1D list of numbers (B,)."
        )
        # check that the first dimension is the batch size
        assert logprobs.size(0) == ctx_length, (
            "Attacker logprobs must have the same batch size as the context."
        )
        # warn if everything is between 0 and 1
        if ((logprobs >= 0.0) & (logprobs <= 1.0)).all():
            logger.warning(
                "Attacker *log*probs looks suspiciously like probabilities, "
                "try taking the .log() of your tensor?"
            )
        self._disable_asserts[check_key] = True

    def _get_attacker_logprobs_and_validate(
        self, context: Sequence[StateT], continuation: Sequence[ActionT]
    ) -> torch.Tensor:
        logprobs = self.get_attacker_logprobs(context, continuation)
        self._check_logprobs("attacker_logprobs", logprobs, len(context), True)
        return logprobs

    def _get_target_logprobs_and_validate(
        self, context: Sequence[StateT], continuation: Sequence[ActionT]
    ) -> torch.Tensor:
        logprobs = self.get_target_logprobs(context, continuation)
        self._check_logprobs("target_logprobs", logprobs, len(context), False)
        return logprobs

    def _get_baseline_logprobs_and_validate(
        self, context: Sequence[StateT], continuation: Sequence[ActionT]
    ) -> torch.Tensor:
        logprobs = self.get_baseline_logprobs(context, continuation)
        self._check_logprobs("baseline_logprobs", logprobs, len(context), False)
        return logprobs

    def _rollout_prompt_with_attacker_and_validate(
        self, x: Sequence[StateT]
    ) -> Sequence[ActionT]:
        rolled_out = self.rollout_prompt_with_attacker(x)
        self._check_continuation("attacker_rollout", x, rolled_out)
        return rolled_out

    def _rollout_prompt_with_target_and_validate(
        self, x: Sequence[StateT]
    ) -> Sequence[StateT]:
        rolled_out = self.rollout_prompt_with_target(x)
        self._check_continuation("target_rollout", x, rolled_out)
        return rolled_out
