"""Tests for device consistency validation in Problem class."""

import pytest
import torch
from typing import Sequence, Iterator

from astra_rl.core.system import System
from astra_rl.core.scorer import Scorer
from tests.utils import mark_gpu


class MockScorer(Scorer[str, str]):
    """Mock scorer for testing."""

    def score(self, x: Sequence[str]) -> Sequence[float]:
        return [0.5] * len(x)


class MockSystem(System[str, str]):
    """Mock system for testing device consistency."""

    def __init__(self, device1: str = "cpu", device2: str = "cpu"):
        super().__init__(MockScorer())
        self.device1 = torch.device(device1)
        self.device2 = torch.device(device2)

    def get_target_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        # Return tensor on device1
        return torch.randn(len(context), 5, device=self.device1)

    def get_baseline_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        # Return tensor on device2 (potentially different)
        return torch.randn(len(context), 5, device=self.device2)

    def get_auditor_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        # Return tensor on device1 with gradients
        return torch.randn(len(context), 5, device=self.device1, requires_grad=True)

    def rollout_prompt_with_auditor(self, x: Sequence[str]) -> Sequence[str]:
        return ["response"] * len(x)

    def rollout_prompt_with_target(self, x: Sequence[str]) -> Sequence[str]:
        return ["response"] * len(x)

    def advance(self, context: str, audit: str, response: str) -> str:
        return context + audit + response

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return iter([])

    def reward(
        self, context: Sequence[str], audit: Sequence[str], response: Sequence[str]
    ) -> Sequence[float]:
        return [0.5] * len(context)


def test_device_consistency_same_device():
    """Test that logprobs on the same device pass validation."""
    system = MockSystem(device1="cpu", device2="cpu")
    context = ["hello", "world"]
    continuation = ["test", "case"]

    # These should all pass since they're on the same device
    auditor_logprobs = system._get_auditor_logprobs_and_validate(context, continuation)
    target_logprobs = system._get_target_logprobs_and_validate(context, continuation)
    baseline_logprobs = system._get_baseline_logprobs_and_validate(
        context, continuation
    )

    assert auditor_logprobs.device == torch.device("cpu")
    assert target_logprobs.device == torch.device("cpu")
    assert baseline_logprobs.device == torch.device("cpu")


@mark_gpu
def test_device_consistency_different_devices_gpu():
    """Test that logprobs on different devices fail validation when GPU is available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("This test requires at least 2 CUDA devices")

    system = MockSystem(device1="cuda:0", device2="cuda:1")
    context = ["hello", "world"]
    continuation = ["test", "case"]

    # First call should succeed and set expected device
    auditor_logprobs = system._get_auditor_logprobs_and_validate(context, continuation)
    assert auditor_logprobs.device == torch.device("cuda:0")

    # Second call with different device should fail
    with pytest.raises(AssertionError) as exc_info:
        system._get_baseline_logprobs_and_validate(context, continuation)

    error_msg = str(exc_info.value)
    assert "All logprobs must be on the same device" in error_msg
    assert "cuda:0" in error_msg
    assert "cuda:1" in error_msg
    assert "baseline_logprobs" in error_msg


def test_device_consistency_different_devices_cpu_mock():
    """Test device consistency validation with mock devices to simulate the error."""
    system = MockSystem(device1="cpu", device2="cpu")
    context = ["hello", "world"]
    continuation = ["test", "case"]

    # Manually set up the scenario by setting expected device first
    system._expected_device = torch.device("cpu")

    # Create a tensor on a "different" device (we'll simulate this by manually changing the expected device)
    target_logprobs = system.get_target_logprobs(context, continuation)

    # Manually change expected device to simulate different device scenario
    system._expected_device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("meta")
    )

    # Now try to validate a CPU tensor when expecting a different device
    with pytest.raises(AssertionError) as exc_info:
        system._check_logprobs("target_logprobs", target_logprobs, len(context), False)

    error_msg = str(exc_info.value)
    assert "All logprobs must be on the same device" in error_msg
    assert "target_logprobs" in error_msg


def test_device_consistency_reset_on_new_problem():
    """Test that device tracking is reset for each new Problem instance."""
    # First problem with CPU
    system1 = MockSystem(device1="cpu", device2="cpu")
    context = ["hello"]
    continuation = ["test"]

    system1._get_auditor_logprobs_and_validate(context, continuation)
    assert system1._expected_device == torch.device("cpu")

    # Second problem should start fresh
    system2 = MockSystem(device1="cpu", device2="cpu")
    assert system2._expected_device is None

    system2._get_target_logprobs_and_validate(context, continuation)
    assert system2._expected_device == torch.device("cpu")


def test_device_consistency_with_disable_asserts():
    """Test that device consistency is only checked once per check_key due to disable_asserts."""
    system = MockSystem(device1="cpu", device2="cpu")
    context = ["hello"]
    continuation = ["test"]

    # First call should set device and disable further checks for this key
    system._get_auditor_logprobs_and_validate(context, continuation)
    assert system._expected_device == torch.device("cpu")
    assert system._disable_asserts["auditor_logprobs"] is True

    # Second call with same key should not perform device check due to disabled asserts
    # We can't easily test this without modifying the tensor device directly,
    # but we can verify the disable mechanism works
    system._get_auditor_logprobs_and_validate(context, continuation)  # Should not raise


def test_error_message_quality():
    """Test that error messages provide helpful information."""
    system = MockSystem(device1="cpu", device2="cpu")

    # Set up a scenario that will trigger the error
    system._expected_device = torch.device("meta")  # Use meta device as "expected"
    cpu_tensor = torch.randn(1, 5)  # CPU tensor

    with pytest.raises(AssertionError) as exc_info:
        system._check_logprobs("test_logprobs", cpu_tensor, 1, False)

    error_msg = str(exc_info.value)
    assert "All logprobs must be on the same device" in error_msg
    assert "Expected meta" in error_msg
    assert "test_logprobs logprobs are on cpu" in error_msg
    assert "models (auditor, target, baseline) are on the same device" in error_msg
