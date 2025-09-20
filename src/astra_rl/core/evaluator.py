"""
evaluator.py
Rollout a problem, and print some desirable metrics.
"""

from abc import abstractmethod, ABC
from dataclass import dataclass
from typing import Dict, Any, List, Union, TypedDict, Generic

from astra_rl.core.common import StateT, ActionT
from astra_rl.core.environment import Environment, Graph

# forward define a "JSON-like" type for metrics, which is
# a nested dict/list structure with str keys and primitive
# values at the leaves.
JSONLike = Union["Nested", "Single"]


class Nested(TypedDict):
    value: Dict[str, JSONLike]


class Single(TypedDict):
    value: Dict[str, Union[str, int, float, bool, None]]


# types of analyses a single graph can produce
@dataclass
class GraphMetrics:
    # metrics for the entire rollout graph
    overall: Dict[str, Any]
    # metrics for each turn, can define "turn" however desired
    per_turn: List[Dict[str, Any]]


# the actual evaluator, which is used to find metrics
class Evaluator(ABC, Generic[StateT, ActionT]):
    """An Evaluator used for rolling out a problem and computing metrics.

    The class takes an `Environment' as input, and calls its `eval_rollout`
    method subject to some seeds which you pass. The resulting graph is
    then analyzed to yield some metrics for printing.
    """

    def __init__(self, env: Environment[StateT, ActionT]):
        self.env = env

    @abstractmethod
    def compute_metrics(self, g: Graph[StateT, ActionT]) -> GraphMetrics:
        """Compute metrics from a rollout graph.

        Args:
        g (Graph[StateT, ActionT]): The rollout graph to analyze.

        Note:


        Returns:
        GraphMetrics: The computed metrics for the rollout graph.
        """

        pass

    @abstractmethod
    def aggregate_metrics(self, all_metrics: List[GraphMetrics]) -> JSONLike:
        """Aggregate metrics across multiple rollout graphs.

        Args:
            all_metrics (List[GraphMetrics]): A list of metrics from multiple rollout graphs.

        Note:
            This method should define how to aggregate the metrics,
            e.g., by averaging or summing them, producing

        Returns:
            JSONLike: The aggregated metrics.
        """

        pass

    @staticmethod
    def write_json(data: JSONLike, path: str) -> None:
        """Write JSON-like data to a file.

        Args:
            data (JSONLike): The data to write.
            path (str): The file path to write the data to.

        Side Effects:
            Writes the data to the specified file in JSON format.
        """
        import json

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
