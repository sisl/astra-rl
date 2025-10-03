"""
detoxify.py
Scorer to call into the Detoxify engine.
"""

from typing import Sequence

from detoxify import Detoxify

from astra_rl.core.scorer import Scorer


class DetoxifyScorer(Scorer[str, str]):
    """Scorer that wraps the Detoxify library for toxicity detection.

    https://github.com/unitaryai/detoxify

    Attributes:
        harm_category (str): The category of harm to detect (default is "toxicity"); see below.
        variant (str): The variant of the Detoxify model to use (default is "original").

    Notes:
        Possible harm categories
        include "toxicity", "severe_toxicity", "obscene", "identity_attack",
        "insult", "threat", "sexual_explicit".

        Possible variants
        Include "original", "multilingual", "unbiased".
    """

    def __init__(self, harm_category: str = "toxicity", variant: str = "original"):
        self.model = Detoxify(variant)
        self.harm_category = harm_category

    def score(self, x: Sequence[str]) -> Sequence[float]:
        # we ignore typing here because we don't actually have the ability
        # to get typing information from detoxify
        return self.model.predict(x)[self.harm_category]  # type: ignore
