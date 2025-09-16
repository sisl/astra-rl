"""
Tests for the Problem class in astra_rl.core.problem
"""

import pytest
import torch
from typing import Sequence, Iterator

from astra_rl.core.problem import Problem
from astra_rl.core.moderator import Moderator


class MockModerator(Moderator[str, str]):
    """Mock moderator for testing."""

    def moderate(self, x: Sequence[str]) -> Sequence[float]:
        # Return a simple score based on string length
        return [len(s) * 0.1 for s in x]


class SimpleProblem(Problem[str, str]):
    """Simple implementation of Problem for testing."""

    def __init__(self, moderator: Moderator[str, str]):
        super().__init__(moderator)
        self._model_params = torch.nn.Parameter(torch.randn(10))

    def get_target_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        # Return mock log probabilities
        batch_size = len(context)
        return torch.randn(batch_size, 1) - 5.0  # Ensure negative values (log probs)

    def get_baseline_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        batch_size = len(context)
        return torch.randn(batch_size, 1) - 4.0

    def get_attacker_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        batch_size = len(context)
        logprobs = torch.randn(batch_size, 1, requires_grad=True) - 3.0
        return logprobs

    def rollout_prompt_with_attacker(self, x: Sequence[str]) -> Sequence[str]:
        return [f"attack_{i}" for i, _ in enumerate(x)]

    def rollout_prompt_with_target(self, x: Sequence[str]) -> Sequence[str]:
        return [f"target_{i}" for i, _ in enumerate(x)]

    def advance(self, context: str, attack: str, response: str) -> str:
        return context + attack + response

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        yield self._model_params

    def reward(
        self, context: Sequence[str], attack: Sequence[str], response: Sequence[str]
    ) -> Sequence[float]:
        return [1.0] * len(context)


class TestProblem:
    """Test suite for the Problem class."""

    @pytest.fixture
    def moderator(self):
        return MockModerator()

    @pytest.fixture
    def problem(self, moderator):
        return SimpleProblem(moderator)

    def test_problem_initialization(self, moderator):
        """Test that Problem can be initialized with a moderator."""

        problem = SimpleProblem(moderator)
        assert problem.moderator is moderator
        assert isinstance(problem._disable_asserts, dict)

    def test_get_target_logprobs(self, problem):
        """Test target logprobs method."""

        context = ["Hello", "World"]
        continuation = ["Goodbye", "World!"]

        logprobs = problem.get_target_logprobs(context, continuation)

        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.shape == (2, 1)
        assert (logprobs < 0).all()  # Should be log probabilities (negative)

    def test_get_baseline_logprobs(self, problem):
        """Test baseline logprobs method."""

        context = ["Hello", "World"]
        continuation = ["Goodbye", "World!"]

        logprobs = problem.get_baseline_logprobs(context, continuation)

        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.shape == (2, 1)
        assert (logprobs < 0).all()

    def test_get_attacker_logprobs(self, problem):
        """Test attacker logprobs method."""
        context = ["Hello", "World"]
        continuation = ["Goodbye", "World!"]

        logprobs = problem.get_attacker_logprobs(context, continuation)

        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.shape == (2, 1)
        assert logprobs.requires_grad
        assert (logprobs < 0).all()

    def test_rollout_prompt_with_attacker(self, problem):
        """Test attacker rollout method."""
        prompts = ["Hello", "How are"]

        result = problem.rollout_prompt_with_attacker(prompts)

        assert len(result) == len(prompts)
        assert result[0] == "attack_0"
        assert result[1] == "attack_1"

    def test_rollout_prompt_with_target(self, problem):
        """Test target rollout method."""
        prompts = ["Hello", "How are"]

        result = problem.rollout_prompt_with_target(prompts)

        assert len(result) == len(prompts)
        assert result[0] == "target_0"
        assert result[1] == "target_1"

    def test_advance(self, problem):
        """Test advance method."""
        context = "Hello"
        attack = " evil"
        response = " good"

        result = problem.advance(context, attack, response)

        assert result == "Hello evil good"

    def test_parameters(self, problem):
        """Test parameters method returns iterator."""
        params = list(problem.parameters())

        assert len(params) == 1
        assert isinstance(params[0], torch.nn.Parameter)
        assert params[0].shape == (10,)

    def test_reward(self, problem):
        """Test reward method."""
        context = ["Hello", "Hello"]
        attack = ["attack1", "attack2"]
        response = ["response1", "response2"]

        rewards = problem.reward(context, attack, response)

        assert len(rewards) == 2
        assert all(r == 1.0 for r in rewards)

    def test_validated_attacker_logprobs(self, problem):
        """Test validation wrapper for attacker logprobs."""
        context = ["Hello"]
        continuation = ["world"]

        logprobs = problem._get_attacker_logprobs_and_validate(context, continuation)

        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.requires_grad
        assert logprobs.shape == (1, 1)

    def test_validated_target_logprobs(self, problem):
        """Test validation wrapper for target logprobs."""
        context = ["Hello"]
        continuation = ["world"]

        logprobs = problem._get_target_logprobs_and_validate(context, continuation)

        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.shape == (1, 1)

    def test_validated_baseline_logprobs(self, problem):
        """Test validation wrapper for baseline logprobs."""
        context = ["Hello"]
        continuation = ["world"]

        logprobs = problem._get_baseline_logprobs_and_validate(context, continuation)

        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.shape == (1, 1)

    def test_validated_attacker_rollout(self, problem):
        """Test validation wrapper for attacker rollout."""
        prompts = ["Hello"]

        result = problem._rollout_prompt_with_attacker_and_validate(prompts)

        assert len(result) == 1
        assert result[0] == "attack_0"

    def test_validated_target_rollout(self, problem):
        """Test validation wrapper for target rollout."""
        prompts = ["Hello"]

        result = problem._rollout_prompt_with_target_and_validate(prompts)

        assert len(result) == 1
        assert result[0] == "target_0"

    def test_check_logprobs_validation(self, problem):
        """Test logprobs validation logic."""
        # Test valid logprobs
        valid_logprobs = torch.tensor([[-1.5], [-2.3]], requires_grad=True)
        problem._check_logprobs("test_valid", valid_logprobs, 2, requires_grad=True)

        # Test invalid logprobs (not a tensor)
        with pytest.raises(AssertionError):
            problem._check_logprobs("test_invalid", [1.0, 2.0], 2, requires_grad=False)

    def test_check_logprobs_grad_requirement(self, problem):
        """Test gradient requirement validation."""
        # Reset assert checking for this test
        problem._disable_asserts["test_grad"] = False

        logprobs_no_grad = torch.tensor([[-1.5], [-2.3]], requires_grad=False)

        with pytest.raises(AssertionError):
            problem._check_logprobs(
                "test_grad", logprobs_no_grad, 2, requires_grad=True
            )

    def test_check_logprobs_batch_size(self, problem):
        """Test batch size validation."""
        # Reset assert checking for this test
        problem._disable_asserts["test_batch"] = False

        wrong_batch_logprobs = torch.tensor([[-1.5]], requires_grad=False)

        with pytest.raises(AssertionError):
            problem._check_logprobs(
                "test_batch", wrong_batch_logprobs, 2, requires_grad=False
            )

    def test_disable_asserts_mechanism(self, problem):
        """Test that assert checking is disabled after first call."""
        # First call should trigger validation
        logprobs = torch.tensor([[-1.5]], requires_grad=True)
        problem._check_logprobs("test_disable", logprobs, 1, requires_grad=True)

        # Assert should now be disabled for this key
        assert problem._disable_asserts["test_disable"]

        # Second call with invalid data should not raise error
        problem._check_logprobs(
            "test_disable", "invalid", 1, requires_grad=True
        )  # Should not raise


class TestProblemAbstractMethods:
    """Test that Problem is properly abstract."""

    def test_cannot_instantiate_abstract_problem(self):
        """Test that Problem cannot be instantiated directly."""
        moderator = MockModerator()

        with pytest.raises(TypeError):
            Problem(moderator)

    def test_abstract_methods_must_be_implemented(self):
        """Test that subclasses must implement all abstract methods."""

        class IncompleteProblem(Problem[str, str]):
            pass

        moderator = MockModerator()

        with pytest.raises(TypeError):
            IncompleteProblem(moderator)


class TestProblemWithWarnings:
    """Test warning cases."""

    @pytest.fixture
    def moderator(self):
        return MockModerator()

    def test_probability_warning(self, moderator, caplog):
        """Test warning when logprobs look like probabilities."""

        class ProbabilityProblem(SimpleProblem):
            def get_attacker_logprobs(
                self, context: Sequence[str], continuation: Sequence[str]
            ) -> torch.Tensor:
                # Return values that look like probabilities (0-1 range)
                batch_size = len(context)
                return torch.rand(batch_size, 1, requires_grad=True)

        problem = ProbabilityProblem(moderator)
        context = ["Hello"]
        continuation = ["world"]

        with caplog.at_level("WARNING"):
            problem._get_attacker_logprobs_and_validate(context, continuation)

        assert "suspiciously like probabilities" in caplog.text
