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
    # Offline path: if LWMF_REPLAY_FILE points to a local JSONL ({"text": ...} per
    # line), read replay docs from it instead of streaming HF. This is what the GPU
    # jobs use (HF_HUB_OFFLINE=1), since FineWeb-Edu can't be streamed offline.
    import os, json
    local = os.environ.get("LWMF_REPLAY_FILE")
    if local:
        with open(local, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get(text_key)
                if text:
                    yield text
        return
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=True)
    for row in ds:
        text = row.get(text_key)
        if text:
            yield text
