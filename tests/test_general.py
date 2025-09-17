import torch

from tests.utils import mark_gpu


@mark_gpu
def test_gpu_available():
    # This test is a simple demonstration of how to mark a test that requires a GPU.
    assert torch.cuda.is_available(), (
        "CUDA is not available. Please check your GPU setup."
    )
    assert torch.cuda.device_count() > 0, "No CUDA devices found."
    print("CUDA is available and at least one GPU is detected.")
