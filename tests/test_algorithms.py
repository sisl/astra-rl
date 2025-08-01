"""
test_algorithms.py

Basic unit tests for ASTRA algorithms module classes and methods.
"""

# Test dpo.py
import pytest
from unittest.mock import patch  # , MagicMock
from astra_rl.algorithms.dpo import DPOStep, DPOBatch, DPO, IPO
from astra_rl.core.environment import Graph, Node
import torch


# Test DPOStep struct
def test_dpo_step_initialization():
    prefix = "state1"
    suffix_pos = "action1"
    suffix_neg = "action2"
    step = DPOStep(prefix=prefix, suffix_pos=suffix_pos, suffix_neg=suffix_neg)

    assert step.prefix == prefix
    assert step.suffix_pos == suffix_pos
    assert step.suffix_neg == suffix_neg


# Test DPOBatch struct
def test_dpo_batch_initialization():
    prefixes = ["state1", "state2"]
    suffix_pos = ["action1", "action2"]
    suffix_neg = ["action3", "action4"]
    batch = DPOBatch(prefixes=prefixes, suffix_pos=suffix_pos, suffix_neg=suffix_neg)

    assert batch.prefixes == prefixes
    assert batch.suffix_pos == suffix_pos
    assert batch.suffix_neg == suffix_neg


# TODO: Do we need wrong attribute type checks? As far as I know, StateT and ActionT could be anything (I just used an int to illustrate what I mean)
# TODO: Also, same question for DPOBatch
# def test_dpo_step_type_validation():
#     with pytest.raises(TypeError):
#         DPOStep(prefix=42, suffix_pos="action1", suffix_neg="action2")


# Test DPO class methods
@pytest.fixture
def mock_problem():
    with patch("astra_rl.core.problem") as mock_problem:
        yield mock_problem


@pytest.fixture
def mock_environment():
    with patch("astra_rl.core.environment") as mock_environment:
        yield mock_environment


def test_dpo_collate_fn(mock_problem):
    steps = [
        DPOStep(prefix="state1", suffix_pos="action1", suffix_neg="action2"),
        DPOStep(prefix="state2", suffix_pos="action3", suffix_neg="action4"),
    ]
    # Initialize DPO with a mock Problem
    dpo = DPO(problem=mock_problem)
    batch = dpo.collate_fn(steps)

    assert batch.prefixes == ["state1", "state2"]
    assert batch.suffix_pos == ["action1", "action3"]
    assert batch.suffix_neg == ["action2", "action4"]


def test_dpo_flatten_method_bad_prefixes(mock_problem):
    node_1 = Node(
        context="context1",
        attack="action1",
        response="response1",
        reward=10.0,
        children=[],
    )
    node_2 = Node(
        context="context2",
        attack="action2",
        response="response2",
        reward=5.0,
        children=[],
    )
    node_3 = Node(
        context="context3",
        attack="action3",
        response="response3",
        reward=2.0,
        children=[node_2, node_2],
    )
    node_sequence = [node_1, node_3]
    graph = Graph(context="state1", children=node_sequence)

    # Initialize DPO with a mock Problem
    dpo = DPO(problem=mock_problem)

    # Error is caused by context values being different (string matching assertion)
    with pytest.raises(AssertionError):
        dpo.flatten(graph)


def test_dpo_flatten_method(mock_problem):
    node_1 = Node(
        context="context1",
        attack="action1",
        response="response1",
        reward=1.0,
        children=[],
    )
    node_2 = Node(
        context="context1",
        attack="action2",
        response="response2",
        reward=5.0,
        children=[],
    )
    node_3 = Node(
        context="context1",
        attack="action3",
        response="response3",
        reward=2.0,
        children=[node_2, node_2],
    )
    node_sequence = [node_1, node_3]
    graph = Graph(context="state1", children=node_sequence)

    # Initialize DPO with a mock Problem
    dpo = DPO(problem=mock_problem)
    result = dpo.flatten(graph)

    assert len(result) == 2
    assert result[0].prefix == "context1"
    # pos should have larger reward than neg, hence action3
    assert result[0].suffix_pos == "action3"
    assert result[0].suffix_neg == "action1"


def test_dpo_step_method(mock_problem):
    # Initialize DPO with a mock Problem
    dpo = DPO(problem=mock_problem, beta=0.1)

    batch = DPOBatch(
        prefixes=["state1", "state2"],
        suffix_pos=["action1", "action3"],
        suffix_neg=["action2", "action4"],
    )

    # Set mock return values for private methods
    # TODO: Are these values ok or should they be different for, e.g., batch.suffix_pos and batch.suffix_neg? Currently they are the same
    mock_problem._get_attacker_logprobs_and_validate.return_value = torch.tensor(
        [1.0, 2.0]
    )
    mock_problem._get_baseline_logprobs_and_validate.return_value = torch.tensor(
        [0.5, 1.5]
    )

    loss, logging_dict = dpo.step(batch)

    assert isinstance(loss, torch.Tensor)
    assert "training/loss" in logging_dict


# test_ipo_step_method (inherits the rest from DPO)
def test_ipo_step_method(mock_problem):
    # Initialize IPO with a mock Problem
    ipo = IPO(problem=mock_problem, beta=0.1)

    batch = DPOBatch(
        prefixes=["state1", "state2"],
        suffix_pos=["action1", "action3"],
        suffix_neg=["action2", "action4"],
    )

    # Set mock return values for private methods
    # TODO: Are these values ok or should they be different for, e.g., batch.suffix_pos and batch.suffix_neg? Currently they are the same
    mock_problem._get_attacker_logprobs_and_validate.return_value = torch.tensor(
        [1.0, 2.0]
    )
    mock_problem._get_baseline_logprobs_and_validate.return_value = torch.tensor(
        [0.5, 1.5]
    )

    loss, logging_dict = ipo.step(batch)

    assert isinstance(loss, torch.Tensor)
    assert "training/loss" in logging_dict
