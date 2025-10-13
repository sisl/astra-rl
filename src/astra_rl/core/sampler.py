"""
sampler.py
Roll out a system, and specify how its sampler behaves.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Sequence, Generic, Self, Optional, Any, Iterator, Union, cast

from astra_rl.core.common import StateT, ActionT
from astra_rl.core.system import System


@dataclass
class Node(Generic[StateT, ActionT]):
    """A node in the rollout graph.

    Represents a single interaction in the rollout process, containing the context,
    the action taken (challenge), the response received, the reward for that action,
    scores from various scorers, and any children nodes that follow this action.

    Attributes:
        context (StateT): The initial state before the action.
        challenge (ActionT): The action taken in this node.
        response (StateT): The resulting state after the action.
        reward (float): The reward received for taking the action.
        scores (dict[str, Any]): Dictionary of scores from various scorers.
        children (Sequence[Node[StateT, ActionT]]): Subsequent nodes that follow this action.
        parent (Node[StateT, ActionT] | None): Parent node in the tree structure.

    Generics:
        StateT (type): The type of the state in the sampler.
        ActionT (type): The type of the action in the sampler.
    """

    context: StateT
    challenge: ActionT
    response: StateT
    reward: float
    scores: dict[str, Any] = field(default_factory=dict)
    children: Sequence[Self] = field(default_factory=list)
    parent: Optional[Self] = field(default=None, repr=False)

    def get_path_from_root(self) -> list[Self]:
        """Get all nodes from root to this node.

        Returns:
            List of nodes from root to this node (inclusive).
        """
        path = []
        current: Optional[Self] = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_conversation_text(self) -> StateT:
        """Get full conversation up to this point.

        Note:
            This assumes StateT supports string concatenation. For str types,
            this concatenates all challenges and responses.

        Returns:
            The full conversation text from root to this node.
        """
        parts: list[Union[StateT, ActionT]] = []
        for node in self.get_path_from_root():
            if node.challenge:
                parts.append(node.challenge)
            parts.append(node.response)
        # Assuming StateT is str or has __add__
        # This method only works when StateT and ActionT are compatible types (e.g., both str)
        result: Any = parts[0] if parts else ""
        for part in parts[1:]:
            result = result + part
        return cast(StateT, result)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children).

        Returns:
            True if this node has no children.
        """
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        """Get the depth of this node in the tree.

        Returns:
            The depth (distance from root, where root has depth 0).
        """
        return len(self.get_path_from_root()) - 1


@dataclass
class Graph(Generic[StateT, ActionT]):
    """A graph representing the rollout (history + actions) of a system.

    Attributes:
        context (StateT): The initial state of the sampler.
        children (Sequence[Node[StateT, ActionT]]): The sequence of root nodes representing actions and responses.
    """

    context: StateT
    children: Sequence[Node[StateT, ActionT]]

    def get_all_trajectories(self) -> list[list[Node[StateT, ActionT]]]:
        """Get all root-to-leaf paths in the graph.

        Returns:
            List of trajectories, where each trajectory is a list of nodes from root to leaf.
        """
        trajectories = []
        for root in self.children:
            for leaf in self._get_leaves(root):
                trajectories.append(leaf.get_path_from_root())
        return trajectories

    def traverse(self) -> Iterator[Node[StateT, ActionT]]:
        """Iterate all nodes depth-first.

        Yields:
            Nodes in depth-first order.
        """
        for root in self.children:
            yield from self._traverse_node(root)

    def get_nodes_at_depth(self, depth: int) -> list[Node[StateT, ActionT]]:
        """Get all nodes at a specific depth in the tree.

        Args:
            depth: The depth level to query (0 = root level).

        Returns:
            List of nodes at the specified depth.
        """
        return [n for n in self.traverse() if n.depth == depth]

    def _get_leaves(self, node: Node[StateT, ActionT]) -> list[Node[StateT, ActionT]]:
        """Get all leaf nodes descending from the given node.

        Args:
            node: The node to start from.

        Returns:
            List of leaf nodes.
        """
        if node.is_leaf():
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaves(child))
        return leaves

    def _traverse_node(
        self, node: Node[StateT, ActionT]
    ) -> Iterator[Node[StateT, ActionT]]:
        """Recursively traverse nodes depth-first.

        Args:
            node: The node to start traversal from.

        Yields:
            Nodes in depth-first order.
        """
        yield node
        for child in node.children:
            yield from self._traverse_node(child)


class Sampler(ABC, Generic[StateT, ActionT]):
    """A Sampler used for rolling out a system.

    The primary point of this class is to make a `Graph` of the system
    by calling the `rollout` method. The sampler can keep/sample
    initial state, but should not have global state that persists
    across rollouts.

    Attributes:
        system (System[StateT, ActionT]): The system instance that defines the sampler and actions.

    Generics:
        StateT (type): The type of the state in the sampler.
        ActionT (type): The type of the action in the sampler.
    """

    def __init__(self, system: System[StateT, ActionT]):
        self.system = system

    @abstractmethod
    def rollout(self, seed: Optional[int] = None) -> Graph[StateT, ActionT]:
        """Roll out a system and return a graph of the actions taken.

        Args:
            seed (Optional[int]): An optional seed; the same seed should produce the same graph.

        Returns:
            Graph[StateT, ActionT]: A graph representing the rollout of the system.
        """

        pass

    def eval_rollout(self, seed: Optional[Any] = None) -> Graph[StateT, ActionT]:
        """Roll out for evaluation, by default just the standard rollout

        Notes:
            This can be customized to whatever the user desires in terms of rollout for eval.
            For instance, for evaluation the seed maybe StateT instead of int since there may
            be another evaluation dataset.

            However, if the seed given is None or an int, a default implementation exists
            which just calls `self.rollout(seed)` and so evaluation can be done without
            needing to override this method.

        Args:
            seed (Optional[Any]): An optional seed; the same seed should produce the same graph.

        Returns:
            Graph[StateT, ActionT]: A graph representing the rollout of the system.
        """

        if seed is None or isinstance(seed, int):
            return self.rollout(seed)

        raise NotImplementedError(
            "eval_rollout not implemented for non-int seeds; please override this method."
        )
