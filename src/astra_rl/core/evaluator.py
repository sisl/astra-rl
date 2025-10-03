"""
evaluator.py
Rollout a problem, and print some desirable metrics.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, Any, List, Union, TypedDict, Generic, TypeVar, Optional

from tqdm import tqdm

from astra_rl.core.common import StateT, ActionT
from astra_rl.core.sampler import Sampler, Graph

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


SeedT = TypeVar("SeedT")


# the actual evaluator, which is used to find metrics
class Evaluator(ABC, Generic[StateT, ActionT, SeedT]):
    """An Evaluator used for rolling out a problem and computing metrics.

    The class takes an `Sampler' as input, and calls its `eval_rollout`
    method subject to some seeds which you pass. The resulting graph is
    then analyzed to yield some metrics for printing.

    Example:
    >>> prompts = (
    ...     GET_PROMPTS()
    ... )
    >>> evaluator = Evaluator(
    ...     env,
    ...     seeds=prompts,
    ... )
    >>> metrics = evaluator.evaluate(
    ...     n_rollouts=10
    ... )
    >>> Evaluator.write_json(
    ...     metrics,
    ...     "metrics.json",
    ... )
    """

    def __init__(
        self, env: Sampler[StateT, ActionT], seeds: Optional[List[SeedT]] = None
    ):
        self.seeds = seeds
        self.env = env

    def evaluate(
        self, n_rollouts: Optional[int] = None, progress: bool = True
    ) -> JSONLike:
        """Performs evaluation by rolling out environment many times

        Args:
            n_rollouts (Optional[int]): The number of rollouts to perform, otherwise all seeds.
            progress (bool): Whether to use tqdm for progress bar.

        Returns:
            JSONLike: The aggregated metrics from all rollouts.
        """

        # use number as deterministic seeds, or just use the provided seeds
        # while checking that everything is the right size
        seeds: List[Any] = []
        if self.seeds is None:
            assert n_rollouts is not None, (
                "Must provide n_rollouts if no seeds are given."
            )
            seeds = list(range(n_rollouts))
        else:
            assert n_rollouts is None or n_rollouts <= len(self.seeds), (
                "n_rollouts cannot be greater than number of seeds."
            )
            seeds = self.seeds[:n_rollouts] if n_rollouts is not None else self.seeds

        # aggergate metrics
        all_metrics: List[GraphMetrics] = []
        iterator = tqdm(seeds) if progress else seeds
        for i in iterator:
            g = self.env.eval_rollout(seed=i)
            metrics = self.compute_metrics(g)
            all_metrics.append(metrics)

        aggregated = self.aggregate_metrics(all_metrics)
        return aggregated

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
