"""Integration test to demonstrate device consistency validation."""
import torch
import pytest
from typing import Sequence, Iterator

from astra_rl.core.problem import Problem
from astra_rl.core.moderator import Moderator
from tests.utils import mark_gpu


class MockModerator(Moderator[str, str]):
    """Mock moderator for testing."""
    
    def moderate(self, x: Sequence[str]) -> Sequence[float]:
        return [0.5] * len(x)


class MultiDeviceProblem(Problem[str, str]):
    """Problem that simulates models on different devices."""
    
    def __init__(self, attacker_device: str, baseline_device: str):
        super().__init__(MockModerator())
        self.attacker_device = torch.device(attacker_device)
        self.baseline_device = torch.device(baseline_device)
    
    def get_target_logprobs(self, context: Sequence[str], continuation: Sequence[str]) -> torch.Tensor:
        return torch.randn(len(context), 5, device=self.baseline_device)
    
    def get_baseline_logprobs(self, context: Sequence[str], continuation: Sequence[str]) -> torch.Tensor:
        return torch.randn(len(context), 5, device=self.baseline_device)
    
    def get_attacker_logprobs(self, context: Sequence[str], continuation: Sequence[str]) -> torch.Tensor:
        return torch.randn(len(context), 5, device=self.attacker_device, requires_grad=True)
    
    def rollout_prompt_with_attacker(self, x: Sequence[str]) -> Sequence[str]:
        return ["response"] * len(x)
    
    def rollout_prompt_with_target(self, x: Sequence[str]) -> Sequence[str]:
        return ["response"] * len(x)
    
    def advance(self, context: str, attack: str, response: str) -> str:
        return context + attack + response
    
    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return iter([])
    
    def reward(self, context: Sequence[str], attack: Sequence[str], response: Sequence[str]) -> Sequence[float]:
        return [0.5] * len(context)


def simulate_dpo_step(problem: MultiDeviceProblem, context: Sequence[str], suffix_pos: Sequence[str], suffix_neg: Sequence[str]):
    """Simulate what happens in DPO.step() method."""
    # This mimics the DPO algorithm's step method
    attacker_logprobs_win = problem._get_attacker_logprobs_and_validate(context, suffix_pos)
    attacker_logprobs_loss = problem._get_attacker_logprobs_and_validate(context, suffix_neg)
    baseline_logprobs_win = problem._get_baseline_logprobs_and_validate(context, suffix_pos)
    baseline_logprobs_loss = problem._get_baseline_logprobs_and_validate(context, suffix_neg)
    
    # Sum per-token logprobs to get sequence logprobs (like DPO does)
    attacker_logprobs_win_sum = attacker_logprobs_win.sum(dim=-1)
    attacker_logprobs_loss_sum = attacker_logprobs_loss.sum(dim=-1)
    baseline_logprobs_win_sum = baseline_logprobs_win.sum(dim=-1)
    baseline_logprobs_loss_sum = baseline_logprobs_loss.sum(dim=-1)
    
    # These operations would fail with cryptic error messages if devices don't match
    pi_logratios = attacker_logprobs_win_sum - attacker_logprobs_loss_sum
    ref_logratios = baseline_logprobs_win_sum - baseline_logprobs_loss_sum
    logits = pi_logratios - ref_logratios
    
    return logits


def test_integration_same_device():
    """Test that DPO-like operations work when all models are on the same device."""
    problem = MultiDeviceProblem(attacker_device="cpu", baseline_device="cpu")
    context = ["hello", "world"]
    suffix_pos = ["good", "response"]
    suffix_neg = ["bad", "response"]
    
    # This should work without any device errors
    logits = simulate_dpo_step(problem, context, suffix_pos, suffix_neg)
    assert logits.device == torch.device("cpu")
    assert logits.shape == (2,)  # Batch size of 2


@mark_gpu
def test_integration_different_devices_fails_early():
    """Test that device mismatch is caught early with clear error message."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("This test requires at least 2 CUDA devices")
    
    problem = MultiDeviceProblem(attacker_device="cuda:0", baseline_device="cuda:1")
    context = ["hello", "world"]
    suffix_pos = ["good", "response"]
    suffix_neg = ["bad", "response"]
    
    # This should fail during validation, not during tensor operations
    with pytest.raises(AssertionError) as exc_info:
        simulate_dpo_step(problem, context, suffix_pos, suffix_neg)
    
    error_msg = str(exc_info.value)
    assert "All logprobs must be on the same device" in error_msg
    assert "cuda:0" in error_msg
    assert "cuda:1" in error_msg
    # The error should mention which type of logprobs failed
    assert "logprobs" in error_msg


def test_integration_cpu_vs_meta_device():
    """Test device mismatch with CPU vs meta device to simulate the error without GPU."""
    problem = MultiDeviceProblem(attacker_device="cpu", baseline_device="meta")
    context = ["hello"]
    suffix_pos = ["good"]
    suffix_neg = ["bad"]
    
    # This should fail during validation
    with pytest.raises(AssertionError) as exc_info:
        simulate_dpo_step(problem, context, suffix_pos, suffix_neg)
    
    error_msg = str(exc_info.value)
    assert "All logprobs must be on the same device" in error_msg
    assert "Expected cpu" in error_msg
    assert "baseline_logprobs" in error_msg  # Since baseline comes after attacker
    assert "meta" in error_msg


def test_integration_error_prevents_cryptic_runtime_error():
    """Test that our device check prevents the cryptic RuntimeError that would occur later."""
    # This test simulates what would happen without our fix
    problem = MultiDeviceProblem(attacker_device="cpu", baseline_device="meta")
    
    # Create tensors manually to show what the cryptic error would look like
    attacker_tensor = torch.randn(2, 5, device="cpu")
    baseline_tensor = torch.randn(2, 5, device="meta")
    
    # This would be the cryptic error that users see without our fix
    with pytest.raises(RuntimeError):
        # This operation would fail with a cryptic message
        _ = attacker_tensor - baseline_tensor
    
    # Now test that our validation provides a much better error message
    context = ["hello", "world"]
    suffix_pos = ["good", "response"]
    
    with pytest.raises(AssertionError) as exc_info:
        simulate_dpo_step(problem, context, suffix_pos, [])
    
    our_error_msg = str(exc_info.value)
    
    # Our error message should be much more helpful
    assert "All logprobs must be on the same device" in our_error_msg
    assert "models (attacker, target, baseline) are on the same device" in our_error_msg
    # Our error is more specific about which component failed
    assert "baseline_logprobs" in our_error_msg or "target_logprobs" in our_error_msg