"""text.py: Text En-/De-coder for Neural Networks"""

# System Imports
from typing import List

# Third-Party Imports
from torch import Tensor, argmax, zeros
from Levenshtein import distance

__author__ = "Johannes Bayer"
__copyright__ = "Copyright 2023, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de"
__status__ = "Prototype"




character_set: List[chr] = []


def set_character_set(new_character_set: List[chr]):

    global character_set
    character_set = new_character_set


def encode_text(text: str, max_text_len: int) -> Tensor:

    global character_set

    steps = max_text_len if max_text_len else len(text)
    encoded_text = zeros(steps, len(character_set))

    for step in range(steps):
        encoded_text[step, character_set.index(text[step] if step < len(text) else " ")] = 1.0

    return encoded_text


def decode_text(encoded_text: Tensor) -> str:

    global character_set
    raw_char_list = (argmax(c).item() for c in encoded_text)
    return "".join((character_set[c] for c in raw_char_list)).strip()


def text_distance(pred: Tensor, target: Tensor) -> float:

    pred_str, target_str = decode_text(pred), decode_text(target)

    return max(len(target_str)-distance(pred_str, target_str), 0) / len(target_str)
