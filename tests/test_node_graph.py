"""Tests for Node and Graph navigation and scoring features."""

from astra_rl.core.sampler import Node, Graph
from astra_rl.core.scorer import Scorer, ScoringMode
from typing import Sequence, Union, Dict, Any


class MockScorer(Scorer[str, str]):
    """Mock scorer for testing."""

    def score(self, x: Sequence[Union[str, str]]) -> Sequence[float]:
        # Return length of string as score
        return [float(len(s)) for s in x]

    def score_node(self, node: Node[str, str]) -> Dict[str, Any]:
        """Override to provide custom scoring logic."""
        if self.mode == ScoringMode.TURN:
            scores = {}
            if node.challenge:
                scores["challenge_length"] = len(node.challenge)
            scores["response_length"] = len(node.response)
            return scores
        elif self.mode == ScoringMode.CUMULATIVE:
            full_text = node.get_conversation_text()
            return {
                "conversation_length": len(full_text),
                "depth": node.depth,
            }
        else:  # FINAL
            if node.is_leaf():
                full_text = node.get_conversation_text()
                return {"final_length": len(full_text)}
            return {}


def test_node_creation():
    """Test creating a basic node."""
    node = Node(
        context="Hello",
        challenge="World",
        response="!",
        reward=1.0,
        scores={},
        children=[],
        parent=None,
    )
    assert node.context == "Hello"
    assert node.challenge == "World"
    assert node.response == "!"
    assert node.reward == 1.0
    assert node.scores == {}
    assert node.children == []
    assert node.parent is None


def test_node_backward_compatibility():
    """Test that utterance property works for backward compatibility."""
    node = Node(
        context="Hello",
        challenge="World",
        response="!",
        reward=1.0,
    )
    # utterance should return challenge
    assert node.utterance == "World"
    assert node.utterance == node.challenge


def test_node_is_leaf():
    """Test is_leaf method."""
    leaf_node = Node(
        context="Hello",
        challenge="World",
        response="!",
        reward=1.0,
    )
    assert leaf_node.is_leaf()

    parent_node = Node(
        context="Hello",
        challenge="World",
        response="!",
        reward=1.0,
        children=[leaf_node],
    )
    assert not parent_node.is_leaf()


def test_node_depth():
    """Test depth property."""
    root = Node(
        context="",
        challenge="A",
        response="B",
        reward=1.0,
        parent=None,
    )
    assert root.depth == 0

    child = Node(
        context="AB",
        challenge="C",
        response="D",
        reward=1.0,
        parent=root,
    )
    assert child.depth == 1

    grandchild = Node(
        context="ABCD",
        challenge="E",
        response="F",
        reward=1.0,
        parent=child,
    )
    assert grandchild.depth == 2


def test_node_get_path_from_root():
    """Test getting path from root."""
    root = Node(
        context="",
        challenge="A",
        response="B",
        reward=1.0,
        parent=None,
    )

    child = Node(
        context="AB",
        challenge="C",
        response="D",
        reward=1.0,
        parent=root,
    )

    grandchild = Node(
        context="ABCD",
        challenge="E",
        response="F",
        reward=1.0,
        parent=child,
    )

    path = grandchild.get_path_from_root()
    assert len(path) == 3
    assert path[0] == root
    assert path[1] == child
    assert path[2] == grandchild


def test_node_get_conversation_text():
    """Test getting conversation text."""
    root = Node(
        context="",
        challenge="Hello",
        response=" World",
        reward=1.0,
        parent=None,
    )

    child = Node(
        context="Hello World",
        challenge="!",
        response="?",
        reward=1.0,
        parent=root,
    )

    # Root conversation: "Hello" + " World"
    assert root.get_conversation_text() == "Hello World"

    # Child conversation: "Hello" + " World" + "!" + "?"
    assert child.get_conversation_text() == "Hello World!?"


def test_graph_traversal():
    """Test graph traversal methods."""
    # Create a simple tree structure
    root1 = Node(
        context="",
        challenge="A",
        response="B",
        reward=1.0,
    )

    child1 = Node(
        context="AB",
        challenge="C",
        response="D",
        reward=1.0,
        parent=root1,
    )

    child2 = Node(
        context="AB",
        challenge="E",
        response="F",
        reward=1.0,
        parent=root1,
    )

    root1.children = [child1, child2]

    graph = Graph(context="", children=[root1])

    # Test traverse
    nodes = list(graph.traverse())
    assert len(nodes) == 3
    assert root1 in nodes
    assert child1 in nodes
    assert child2 in nodes


def test_graph_get_all_trajectories():
    """Test getting all trajectories from graph."""
    root = Node(
        context="",
        challenge="A",
        response="B",
        reward=1.0,
    )

    child1 = Node(
        context="AB",
        challenge="C",
        response="D",
        reward=1.0,
        parent=root,
    )

    child2 = Node(
        context="AB",
        challenge="E",
        response="F",
        reward=1.0,
        parent=root,
    )

    root.children = [child1, child2]

    graph = Graph(context="", children=[root])

    trajectories = graph.get_all_trajectories()
    assert len(trajectories) == 2
    assert len(trajectories[0]) == 2  # root -> child1
    assert len(trajectories[1]) == 2  # root -> child2
    assert trajectories[0][0] == root
    assert trajectories[0][1] == child1
    assert trajectories[1][0] == root
    assert trajectories[1][1] == child2


def test_graph_get_nodes_at_depth():
    """Test getting nodes at specific depth."""
    root = Node(
        context="",
        challenge="A",
        response="B",
        reward=1.0,
    )

    child1 = Node(
        context="AB",
        challenge="C",
        response="D",
        reward=1.0,
        parent=root,
    )

    child2 = Node(
        context="AB",
        challenge="E",
        response="F",
        reward=1.0,
        parent=root,
    )

    grandchild = Node(
        context="ABCD",
        challenge="G",
        response="H",
        reward=1.0,
        parent=child1,
    )

    root.children = [child1, child2]
    child1.children = [grandchild]

    graph = Graph(context="", children=[root])

    depth_0 = graph.get_nodes_at_depth(0)
    assert len(depth_0) == 1
    assert root in depth_0

    depth_1 = graph.get_nodes_at_depth(1)
    assert len(depth_1) == 2
    assert child1 in depth_1
    assert child2 in depth_1

    depth_2 = graph.get_nodes_at_depth(2)
    assert len(depth_2) == 1
    assert grandchild in depth_2


def test_scorer_modes():
    """Test scorer with different modes."""
    node = Node(
        context="",
        challenge="Hello",
        response="World",
        reward=1.0,
    )

    # TURN mode
    turn_scorer = MockScorer(mode=ScoringMode.TURN)
    turn_scores = turn_scorer.score_node(node)
    assert "challenge_length" in turn_scores
    assert "response_length" in turn_scores
    assert turn_scores["challenge_length"] == 5
    assert turn_scores["response_length"] == 5

    # CUMULATIVE mode
    cumulative_scorer = MockScorer(mode=ScoringMode.CUMULATIVE)
    cumulative_scores = cumulative_scorer.score_node(node)
    assert "conversation_length" in cumulative_scores
    assert "depth" in cumulative_scores
    assert cumulative_scores["conversation_length"] == 10  # "HelloWorld"
    assert cumulative_scores["depth"] == 0

    # FINAL mode (leaf node)
    final_scorer = MockScorer(mode=ScoringMode.FINAL)
    final_scores = final_scorer.score_node(node)
    assert "final_length" in final_scores
    assert final_scores["final_length"] == 10


def test_node_scores_dict():
    """Test that scores can be stored and retrieved from node."""
    node = Node(
        context="",
        challenge="Hello",
        response="World",
        reward=1.0,
        scores={"toxicity": 0.5, "perplexity": 10.0},
    )

    assert node.scores["toxicity"] == 0.5
    assert node.scores["perplexity"] == 10.0

    # Test adding scores
    node.scores["new_score"] = 42.0
    assert node.scores["new_score"] == 42.0
