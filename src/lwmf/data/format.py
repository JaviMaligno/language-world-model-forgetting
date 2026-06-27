from __future__ import annotations
from dataclasses import dataclass
from lwmf.schema import Trajectory

@dataclass(frozen=True)
class TurnExample:
    prefix_messages: list[dict]
    target: str

def trajectory_to_messages(traj: Trajectory) -> list[dict]:
    msgs = [{"role": "system", "content": traj.system}]
    for t in traj.turns:
        msgs.append({"role": "user", "content": t.action})
        msgs.append({"role": "assistant", "content": t.observation})
    return msgs

def expand_to_turns(traj: Trajectory) -> list[TurnExample]:
    examples: list[TurnExample] = []
    prefix = [{"role": "system", "content": traj.system}]
    for t in traj.turns:
        ex_prefix = prefix + [{"role": "user", "content": t.action}]
        examples.append(TurnExample(prefix_messages=ex_prefix, target=t.observation))
        # resolve this turn into the running prefix for the next example
        prefix = ex_prefix + [{"role": "assistant", "content": t.observation}]
    return examples
