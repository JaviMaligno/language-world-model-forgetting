from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Iterable

@dataclass(frozen=True)
class Turn:
    action: str
    observation: str

@dataclass(frozen=True)
class Trajectory:
    scenario: str
    system: str
    turns: list[Turn]

    def to_json(self) -> str:
        return json.dumps(
            {"scenario": self.scenario, "system": self.system,
             "turns": [asdict(t) for t in self.turns]},
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(s: str) -> "Trajectory":
        d = json.loads(s)
        return Trajectory(d["scenario"], d["system"],
                          [Turn(**t) for t in d["turns"]])

def write_jsonl(path: str, trajs: Iterable[Trajectory]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(t.to_json() + "\n")

def read_jsonl(path: str) -> list[Trajectory]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Trajectory.from_json(line))
    return out
