from __future__ import annotations
import random
from typing import Iterator

def mixed_iterator(narrow: list, replay: Iterator, ratio: float, seed: int) -> Iterator:
    rng = random.Random(seed)
    i = 0
    n = len(narrow)
    while True:
        if ratio > 0.0 and rng.random() < ratio:
            yield next(replay)
        else:
            yield narrow[i % n]
            i += 1

def load_replay_stream(name: str = "HuggingFaceFW/fineweb-edu",
                       split: str = "train", text_key: str = "text") -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=True)
    for row in ds:
        text = row.get(text_key)
        if text:
            yield text
