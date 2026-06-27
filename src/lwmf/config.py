from __future__ import annotations
from dataclasses import dataclass, fields
import yaml

@dataclass
class TrainConfig:
    name: str
    base_model: str
    method: str
    mixing_ratio: float
    lr: float
    max_steps: int
    seq_len: int = 1024
    batch_size: int = 4
    grad_accum: int = 4
    seed: int = 0
    replay_dataset: str = "HuggingFaceFW/fineweb-edu"
    is_instruct: bool = False

    def validate(self) -> None:
        if self.method not in {"full", "lora"}:
            raise ValueError(f"bad method: {self.method}")
        if not 0.0 <= self.mixing_ratio <= 1.0:
            raise ValueError(f"bad mixing_ratio: {self.mixing_ratio}")

def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    allowed = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in d.items() if k in allowed})
