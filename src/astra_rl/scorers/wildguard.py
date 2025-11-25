"""
wildguard.py
Scorer to call into the Wildguard engine.
"""

from typing import Sequence
from astra_rl.core.scorer import Scorer
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM

VALID_SCORING_TARGETS = {
    "harmul_request",
    "response_refusal",
    "harmful_response",
}


class WildguardScorer(Scorer[str, str]):
    """Scorer that uses the Wildguard toxicity classfier. 

    https://github.com/allenai/wildguard

    Attributes:
        scoring_target (str): The category of harm to detect. Must be one of:
            - "harmful_request"
            - "response_refusal"
            - "harmful_response"
            Default is "harmful_response".

        variant (str): The variant of the Detoxify model to use.
            Default is "original".


    Notes:
        Possible harm categories
        include "toxicity", "severe_toxicity", "obscene", "identity_attack",
        "insult", "threat", "sexual_explicit".

        Possible variants
        Include "original", "multilingual", "unbiased".
    """

    def __init__(self, scoring_target: str="harmful_response", variant: str = "original"):
        if scoring_target not in VALID_SCORING_TARGETS:
            raise ValueError(f"scoring_target '{scoring_target}' is not a valid scoring target. Select one of the following: {VALID_SCORING_TARGETS}")
        
        self.model = None
        self.scoring_target = scoring_target

    def score(self, x: Sequence[str]) -> Sequence[float]:
        return 0