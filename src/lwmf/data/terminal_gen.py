from __future__ import annotations
import os
import random
import shutil
from lwmf.data.sandbox import run_in_sandbox
from lwmf.schema import Trajectory, Turn, write_jsonl

SYSTEM_PROMPT = (
    "You are simulating a Linux terminal. Given a shell command, respond with "
    "the exact stdout/stderr the terminal would produce, followed by an exit "
    "marker like [exit 0]. Do not add explanation."
)

# Command templates that produce deterministic, capturable output.
def build_scenario(rng: random.Random) -> list[str]:
    fname = rng.choice(["a.txt", "notes.md", "data.csv"])
    word = rng.choice(["hello", "world", "alpha", "beta"])
    n = rng.randint(1, 3)
    blocks: list[list[str]] = [
        [f"echo {word} > {fname}", f"cat {fname}", "ls -1"],
        [f"mkdir -p d{n}", f"cd d{n} && pwd", "ls -a"],
        [f"printf '%s\\n%s\\n' {word} {word}{n} > {fname}", f"wc -l {fname}", f"head -1 {fname}"],
        ["python3 -c 'print(2+2)'", f"python3 -c 'print(len(\"{word}\"))'"],
        [f"echo {word} >> log", "cat log", "rm log", "ls -1"],
    ]
    chosen = rng.sample(blocks, k=min(n + 1, len(blocks)))
    cmds: list[str] = []
    for b in chosen:
        cmds.extend(b)
    return cmds

# Harder, out-of-distribution commands NOT used in build_scenario (sed/awk/grep/
# sort/tr/cut/rev/uniq + pipes). Deterministic output. Used to build a held-out set
# that probes generalization beyond the training distribution (does full-FT learn the
# terminal task more deeply than LoRA?).
def build_scenario_hard(rng: random.Random) -> list[str]:
    word = rng.choice(["apple", "banana", "cherry", "delta"])
    n = rng.randint(2, 3)
    blocks: list[list[str]] = [
        [f"printf 'c\\nb\\na\\n' | sort", f"echo {word} | tr a-z A-Z"],
        [f"printf 'x y z\\n' | cut -d' ' -f2", f"echo {word} | rev"],
        [f"printf 'a\\na\\nb\\n' | uniq -c", f"echo {word} | sed 's/a/X/g'"],
        [f"printf '1 2\\n3 4\\n' | awk '{{print $2}}'", f"printf 'apple\\nbanana\\n' | grep a"],
        [f"echo {word} | wc -c", f"printf '%s\\n%s\\n' {word} {word} | sort -u"],
    ]
    chosen = rng.sample(blocks, k=min(n, len(blocks)))
    cmds: list[str] = []
    for b in chosen:
        cmds.extend(b)
    return cmds

def generate_trajectories(out_path: str, n_scenarios: int, seed: int,
                          scratch_root: str, scenario_fn=build_scenario) -> int:
    rng = random.Random(seed)
    os.makedirs(scratch_root, exist_ok=True)
    trajs: list[Trajectory] = []
    for i in range(n_scenarios):
        cmds = scenario_fn(rng)
        box = os.path.join(scratch_root, f"s{i}")
        os.makedirs(box, exist_ok=True)
        pairs = run_in_sandbox(cmds, box)
        shutil.rmtree(box, ignore_errors=True)
        turns = [Turn(action=c, observation=o) for c, o in pairs]
        trajs.append(Trajectory(scenario="terminal", system=SYSTEM_PROMPT, turns=turns))
    write_jsonl(out_path, trajs)
    return len(trajs)
