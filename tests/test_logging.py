"""
test_logging.py

Basic unit tests for ASTRA logging module classes and methods.
"""

# Test logging.py

import pytest
import os
from unittest.mock import patch, MagicMock
from astra_rl.logging import logger, ASTRAWandbLogger


# Test build in logger initialization and simple call with string text
# TODO: Do we need to test this further?
def test_standard_logger(caplog):
    logger.info("This is a test log message.")
    assert "This is a test log message." in caplog.text


# Test ASTRAWandbLogger initialization and logging functionality (using mock wandb package)
@pytest.fixture
def mock_wandb():
    with patch("astra_rl.logging.wandb") as mock_wandb:
        yield mock_wandb


@pytest.fixture
def set_env_variable():
    os.environ["WANDB_API_KEY"] = "test_api_key"
    yield
    del os.environ["WANDB_API_KEY"]


def test_logger_init_without_wandb_installed():
    with patch("astra_rl.logging.is_wandb_installed", False):
        with pytest.raises(ImportError, match="Wandb not installed."):
            ASTRAWandbLogger(wandb_kwargs={})


def test_logger_init_without_api_key(mock_wandb):
    if "WANDB_API_KEY" in os.environ:
        del os.environ["WANDB_API_KEY"]
    with pytest.raises(
        EnvironmentError, match="WANDB_API_KEY environment variable is not set."
    ):
        ASTRAWandbLogger(wandb_kwargs={})


def test_logger_init_success(mock_wandb, set_env_variable):
    mock_wandb.init.return_value = MagicMock()
    wandb_logger = ASTRAWandbLogger(wandb_kwargs={"key": "value"})
    mock_wandb.init.assert_called_once_with(project="astra_rl", config={"key": "value"})
    assert wandb_logger.run == mock_wandb.init.return_value


def test_logger_log(mock_wandb, set_env_variable):
    mock_wandb.init.return_value = MagicMock()
    wandb_logger = ASTRAWandbLogger(wandb_kwargs={"key": "value"})
    mock_logs = {"step": 1, "accuracy": 0.95}
    wandb_logger.log(mock_logs)
    wandb_logger.run.log.assert_called_once_with(mock_logs)
