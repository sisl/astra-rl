"""
A "Problem" is one of the core abstractions in Astra RL, defining how to interact
with the system under test. The interface is defined by the `Problem` class, which
defines a set of abstract methods that users must implement to create a custom problem.
This provides flexibility in terms of how users can define their own applications
while still adhering to a common interface that enables the Astra RL framework
to function correctly.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence, Dict, Generic, Union, Iterator, Optional

import torch

from astra_rl.utils import logger
from astra_rl.core.moderator import Moderator
from astra_rl.core.common import StateT, ActionT


class Problem(ABC, Generic[StateT, ActionT]):
    """Defines the core problem interface for Astra RL.

    This class is responsible for defining how exactly to interact
    with the system under test---with generics in terms of how to get
    probabilities and rollouts from the attacker and target models.

    This allows for us to be generic over the types of states, actions
    as well as how to measure them. We ask for a moderator as a way to
    ensure that subclasses can all be generic over the exact metric, and
    instead can only be opinonated about how to achieve the metric.

    Attributes:
        moderator (Moderator[StateT, ActionT]): The moderator used to evaluate sequences.

    Generics:
        StateT (type): The type of the state in the environment.
        ActionT (type): The type of the action in the environment.
    """

    def __init__(self, moderator: Moderator[StateT, ActionT]) -> None:
        # we check all asserts once, and then disable them
        self._disable_asserts: Dict[str, bool] = defaultdict(bool)
        # track the device of the first logprobs tensor to ensure consistency
        self._expected_device: Optional[torch.device] = None
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
            torch.Tensor: The per-token log probabilities of the continuations given their contexts.
                         Shape: (batch_size, max_continuation_length)
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
            torch.Tensor: The per-token log probabilities of the continuations given their contexts.
                         Shape: (batch_size, max_continuation_length)
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
            torch.Tensor: The per-token log probabilities of the continuations given their contexts.
                         Shape: (batch_size, max_continuation_length)
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
        assert isinstance(logprobs, torch.Tensor), "Logprobs must be a torch.Tensor."
        if requires_grad:
            assert logprobs.requires_grad, (
                "Attacker logprobs must carry gradient information."
            )
        # check that the size of the tensor is B x T, where B is the batch size and T is max_continuation_length
        assert logprobs.dim() == 2, (
            "Logprobs must be a 2D tensor (batch_size, max_continuation_length)."
        )
        # check that the first dimension is the batch size
        assert logprobs.size(0) == ctx_length, (
            "Logprobs must have the same batch size as the context."
        )
        # check device consistency across all logprobs
        if self._expected_device is None:
            # This is the first logprobs tensor we've seen, set the expected device
            self._expected_device = logprobs.device
        else:
            # Validate that this tensor is on the same device as previous ones
            assert logprobs.device == self._expected_device, (
                f"All logprobs must be on the same device. Expected {self._expected_device}, "
                f"but {check_key} logprobs are on {logprobs.device}. "
                f"This typically happens when models are on different devices. "
                f"Please ensure all models (attacker, target, baseline) are on the same device."
            )
        # warn if everything is between 0 and 1
        if ((logprobs >= 0.0) & (logprobs <= 1.0)).all():
            logger.warning(
                "Logprobs looks suspiciously like probabilities, "
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


class ValueFunctionProblem(Problem[StateT, ActionT], ABC):
    """Extends `Problem` to be able to return sequence values with a value head.

    Note:
        This is useful for value-laiden solution methods such as Actor
        Critic derivatives (i.e., PPO).

    Attributes:
        moderator (Moderator[StateT, ActionT]): The moderator used to evaluate sequences.

    Generics:
        StateT (type): The type of the state in the environment.
        ActionT (type): The type of the action in the environment.
    """

    @abstractmethod
    def value(
        self, context: Sequence[StateT], continuation: Sequence[ActionT]
    ) -> torch.Tensor:
        """Given a squence, evaluate its token-wise value using a value function.

        Notes:
           This is typically done by the same neural network you use for rollouts
           just passing the intermediate activations through another layer.

        Args:
            context: The context sequence.
            continuation: The continuation sequence to evaluate.

        Returns:
            torch.Tensor[batch_size, max_continuation_length]: The per-token values of
            the given squence by the sequence predictor. Do not include the value of the input
            prefixes. If you are predicting on the whole input, you should be slicing on
            `[:, :-1]`, meaning you should *not* return the value of the last token, whose
            input is eos/context length limit.
        """

        pass
