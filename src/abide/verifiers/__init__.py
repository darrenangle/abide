"""
Verifiers framework integration.

This module provides compatibility with the verifiers framework
for using abide constraints as reward functions in RL training.

Example:
    >>> from abide.verifiers import PoeticFormReward
    >>> from abide.forms import Haiku
    >>>
    >>> reward_fn = PoeticFormReward(Haiku())
    >>> score = reward_fn("An old silent pond\\nA frog jumps in\\nSplash!")
"""

from abide.verifiers.evals import (
    AbideMajorPoeticForms,
    make_poetic_forms_eval,
)
from abide.verifiers.reward import PoeticFormReward, make_reward_function

__all__ = [
    "AbideMajorPoeticForms",
    "PoeticFormReward",
    "make_poetic_forms_eval",
    "make_reward_function",
]
