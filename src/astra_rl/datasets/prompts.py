import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AdvBenchPrompt:
    """A prompt from the AdvBench dataset.

    Attributes:
        goal: The harmful behavior goal.
        target: The target output that the model should produce.
    """

    goal: str
    target: str


def _load_adv_bench_prompts() -> list[AdvBenchPrompt]:
    """Load AdvBench prompts from CSV file."""
    data_path = Path(__file__).parent.parent / "data" / "adv_bench_prompts.csv"
    prompts = []

    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(AdvBenchPrompt(goal=row["goal"], target=row["target"]))

    return prompts


ADV_BENCH_PROMPTS = _load_adv_bench_prompts()
"""List of AdvBench prompts.

This dataset contains a list of `AdvBenchPrompt` objects containing the `goal` and `target`
for each prompt.

It is sourced from the [AdvBench dataset](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)
"""


def _load_task_prompts() -> list[str]:
    """Load generic task prompts from JSON file."""
    data_path = Path(__file__).parent.parent / "data" / "task_prompts.json"
    with open(data_path, "r") as f:
        return json.load(f)  # type: ignore[no-any-return]


ADV_BENCH_BASE_TASK_PROMPTS = _load_task_prompts()
"""List of generic task prompts extracted from AdvBench.

These are generic tasks without the harmful context, e.g. "Write a script" or "Provide a recipe".
"""


def _load_convokit_prompts(split: str) -> list[str]:
    """Load ConvoKit Reddit prompts from JSON file.

    Args:
        split: Dataset split - 'train', 'dev', or 'test'
    """
    data_path = (
        Path(__file__).parent.parent / "data" / f"convokit_reddit_prompts_{split}.json"
    )
    with open(data_path, "r") as f:
        return json.load(f)  # type: ignore[no-any-return]


CONVOKIT_REDDIT_TRAIN = _load_convokit_prompts("train")
"""List of Reddit conversation prompts for training.

This dataset contains prompts extracted from Reddit conversations using ConvoKit.
Prompts have been pre-filtered with Detoxify to be neutral/non-toxic.
Train split with ~2,476 prompts.
"""

CONVOKIT_REDDIT_DEV = _load_convokit_prompts("dev")
"""List of Reddit conversation prompts for development/validation.

This dataset contains prompts extracted from Reddit conversations using ConvoKit.
Prompts have been pre-filtered with Detoxify to be neutral/non-toxic.
Dev split with ~154 prompts.
"""

CONVOKIT_REDDIT_TEST = _load_convokit_prompts("test")
"""List of Reddit conversation prompts for testing.

This dataset contains prompts extracted from Reddit conversations using ConvoKit.
Prompts have been pre-filtered with Detoxify to be neutral/non-toxic.
Test split with ~465 prompts.
"""
