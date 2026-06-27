from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict

@dataclass
class ResultRecord:
    config: dict
    before: dict
    after: dict
    sim_before: dict
    sim_after: dict
    meta: dict

def deltas(record: ResultRecord) -> dict[str, float]:
    d = {task: record.after[task] - record.before[task]
         for task in record.before if task in record.after}
    d["sim_em_gain"] = record.sim_after.get("sim_em", 0.0) - record.sim_before.get("sim_em", 0.0)
    return d

def save_record(record: ResultRecord, results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    name = record.config.get("name", "run")
    seed = record.config.get("seed", 0)
    path = os.path.join(results_dir, f"{name}-seed{seed}.json")
    with open(path, "w") as f:
        json.dump(asdict(record), f, indent=2)
    return path

def append_experiments_md(record: ResultRecord, d: dict, md_path: str) -> None:
    c = record.config
    row = (f"| {c.get('name')} | {c.get('mixing_ratio')} | {c.get('method')} | "
           f"{c.get('seed')} | {record.sim_after.get('sim_em'):.3f} | "
           f"{d.get('mmlu', float('nan')):+.3f} | {d.get('triviaqa', float('nan')):+.3f} |\n")
    header = ("| run | mix | method | seed | sim_em | Δmmlu | Δtrivia |\n"
              "|---|---|---|---|---|---|---|\n")
    exists = os.path.exists(md_path)
    with open(md_path, "a") as f:
        if not exists:
            f.write("# Experiments\n\n" + header)
        f.write(row)
