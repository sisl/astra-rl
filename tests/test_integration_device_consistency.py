"""Integration test to demonstrate device consistency validation."""

import torch
import pytest
from typing import Sequence, Iterator

from astra_rl.core.system import TrainableSystem
from astra_rl.core.scorer import Scorer
from tests.utils import mark_gpu


class MockScorer(Scorer[str, str]):
    """Mock scorer for testing."""

    def score(self, x: Sequence[str]) -> Sequence[float]:
        return [0.5] * len(x)


class MultiDeviceSystem(TrainableSystem[str, str]):
    """System that simulates models on different devices."""

    def __init__(self, tester_device: str, baseline_device: str):
        super().__init__()
        self.tester_device = torch.device(tester_device)
        self.baseline_device = torch.device(baseline_device)

    def get_target_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        return torch.randn(len(context), 5, device=self.baseline_device)

    def get_baseline_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        return torch.randn(len(context), 5, device=self.baseline_device)

    def get_tester_logprobs(
        self, context: Sequence[str], continuation: Sequence[str]
    ) -> torch.Tensor:
        return torch.randn(
            len(context), 5, device=self.tester_device, requires_grad=True
        )

    def rollout_prompt_with_tester(self, x: Sequence[str]) -> Sequence[str]:
        return ["response"] * len(x)

    def rollout_prompt_with_target(self, x: Sequence[str]) -> Sequence[str]:
        return ["response"] * len(x)

    def advance(self, context: str, action: str | None, response: str) -> str:
        if action is None:
            return context + response
        return context + action + response

    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return iter([])

    def reward(
        self,
        context: Sequence[str],
        challenge: Sequence[str | None],
        response: Sequence[str],
    ) -> Sequence[float]:
        return [0.5] * len(context)


def simulate_dpo_step(
    system: MultiDeviceSystem,
    context: Sequence[str],
    suffix_pos: Sequence[str],
    suffix_neg: Sequence[str],
):
    """Simulate what happens in DPO.step() method."""
    # This mimics the DPO algorithm's step method
    tester_logprobs_win = system._get_tester_logprobs_and_validate(context, suffix_pos)
    tester_logprobs_loss = system._get_tester_logprobs_and_validate(context, suffix_neg)
    baseline_logprobs_win = system._get_baseline_logprobs_and_validate(
        context, suffix_pos
    )
    baseline_logprobs_loss = system._get_baseline_logprobs_and_validate(
        context, suffix_neg
    )

    # Sum per-token logprobs to get sequence logprobs (like DPO does)
    tester_logprobs_win_sum = tester_logprobs_win.sum(dim=-1)
    tester_logprobs_loss_sum = tester_logprobs_loss.sum(dim=-1)
    baseline_logprobs_win_sum = baseline_logprobs_win.sum(dim=-1)
    baseline_logprobs_loss_sum = baseline_logprobs_loss.sum(dim=-1)

    # These operations would fail with cryptic error messages if devices don't match
    pi_logratios = tester_logprobs_win_sum - tester_logprobs_loss_sum
    ref_logratios = baseline_logprobs_win_sum - baseline_logprobs_loss_sum
    logits = pi_logratios - ref_logratios

    return logits


def test_integration_same_device():
    """Test that DPO-like operations work when all models are on the same device."""
    system = MultiDeviceSystem(tester_device="cpu", baseline_device="cpu")
    context = ["hello", "world"]
    suffix_pos = ["good", "response"]
    suffix_neg = ["bad", "response"]

    # This should work without any device errors
    logits = simulate_dpo_step(system, context, suffix_pos, suffix_neg)
    assert logits.device == torch.device("cpu")
    assert logits.shape == (2,)  # Batch size of 2


@mark_gpu
def test_integration_different_devices_fails_early():
    """Test that device mismatch is caught early with clear error message."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("This test requires at least 2 CUDA devices")

    system = MultiDeviceSystem(tester_device="cuda:0", baseline_device="cuda:1")
    context = ["hello", "world"]
    suffix_pos = ["good", "response"]
    suffix_neg = ["bad", "response"]

    # This should fail during validation, not during tensor operations
    with pytest.raises(AssertionError) as exc_info:
        simulate_dpo_step(system, context, suffix_pos, suffix_neg)

    error_msg = str(exc_info.value)
    assert "All logprobs must be on the same device" in error_msg
    assert "cuda:0" in error_msg
    assert "cuda:1" in error_msg
    # The error should mention which type of logprobs failed
    assert "logprobs" in error_msg


def test_integration_cpu_vs_meta_device():
    """Test device mismatch with CPU vs meta device to simulate the error without GPU."""
    system = MultiDeviceSystem(tester_device="cpu", baseline_device="meta")
    context = ["hello"]
    suffix_pos = ["good"]
    suffix_neg = ["bad"]

    # This should fail during validation
    with pytest.raises(AssertionError) as exc_info:
        simulate_dpo_step(system, context, suffix_pos, suffix_neg)

    error_msg = str(exc_info.value)
    assert "All logprobs must be on the same device" in error_msg
    assert "Expected cpu" in error_msg
    assert "baseline_logprobs" in error_msg  # Since baseline comes after tester
    assert "meta" in error_msg


def test_integration_error_prevents_cryptic_runtime_error():
    """Test that our device check prevents the cryptic RuntimeError that would occur later."""
    # This test simulates what would happen without our fix
    system = MultiDeviceSystem(tester_device="cpu", baseline_device="meta")

    # Create tensors manually to show what the cryptic error would look like
    tester_tensor = torch.randn(2, 5, device="cpu")
    baseline_tensor = torch.randn(2, 5, device="meta")

    # This would be the cryptic error that users see without our fix
    with pytest.raises(RuntimeError):
        # This operation would fail with a cryptic message
        _ = tester_tensor - baseline_tensor

    # Now test that our validation provides a much better error message
    context = ["hello", "world"]
    suffix_pos = ["good", "response"]

    with pytest.raises(AssertionError) as exc_info:
        simulate_dpo_step(system, context, suffix_pos, [])

    our_error_msg = str(exc_info.value)

    # Our error message should be much more helpful
    assert "All logprobs must be on the same device" in our_error_msg
    assert "models (tester, target, baseline) are on the same device" in our_error_msg
    # Our error is more specific about which component failed
    assert "baseline_logprobs" in our_error_msg or "target_logprobs" in our_error_msg
