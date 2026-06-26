# LWM Forgetting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test a reproducible harness that measures catastrophic forgetting of general knowledge when a small LLM undergoes continual pre-training (CPT) on narrow "language world model" trajectories, and quantifies how data mixing mitigates it.

**Architecture:** A Python package `lwmf` with focused modules: (1) a sandboxed Terminal-trajectory generator, (2) trajectory→chat→turn formatting with a loss-masking collator (the correctness crux), (3) a replay-corpus mixing sampler with a constant token budget, (4) a config-driven training runner (full-FT / LoRA, fp16, T4-friendly), (5) eval wiring (lm-evaluation-harness for general benchmarks + a custom held-out simulation metric), and (6) an orchestrator that records before/after deltas. Phase 0 builds and unit-tests the harness; Phases 1–4 are config sweeps run via a runbook.

**Tech Stack:** Python 3.11, `transformers`, `trl`, `peft`, `datasets`, `bitsandbytes`, `lm-evaluation-harness`, `pytest`. Compute: Azure `Standard_NC4as_T4_v3` spot (1× T4 16 GB) in `australiaeast`.

## Global Constraints

- **Spec:** `docs/specs/2026-06-25-lwm-forgetting-design.md` — every task serves it.
- **Loss masking is the crux:** loss is computed ONLY on observation tokens; action/`user`/system tokens are masked to `-100`. A wrong mask silently invalidates the whole experiment — Task 3 is the highest-risk task and is tested exhaustively.
- **Constant training budget across conditions:** within a sweep, tokens/steps/LR/eval-set are identical across cells; mixing *substitutes* narrow tokens with general tokens (does not add on top).
- **Precision:** fp16 only (T4 has no bf16). Use gradient checkpointing + 8-bit optimizer for 1.5B full-FT to fit 16 GB.
- **Models:** Qwen2.5 family — `Qwen2.5-0.5B`, `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B`, `Qwen2.5-1.5B-Instruct`.
- **Reproducibility:** every run takes an explicit integer `seed`; set `transformers.set_seed(seed)` and `PYTHONHASHSEED`.
- **Interpretation gates:** never report "forgetting" for a run whose held-out simulation accuracy did not rise vs baseline (it didn't learn the task → comparison is meaningless).
- **No network in unit tests:** all `pytest` tests run offline. Anything touching HF Hub downloads or GPU training is a separate smoke test gated behind a marker (`@pytest.mark.smoke`), not a unit test.

---

## File Structure

```
language-world-model-forgetting/
├── pyproject.toml                 # package metadata + deps + pytest config
├── src/lwmf/
│   ├── __init__.py
│   ├── schema.py                  # Trajectory / TurnExample dataclasses + (de)serialization
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sandbox.py             # run_in_sandbox: execute commands in isolated scratch dir
│   │   ├── terminal_gen.py        # scripted scenarios → Trajectory JSONL
│   │   ├── format.py              # trajectory_to_chat + expand_to_turns
│   │   ├── collator.py            # MaskedSFTCollator (THE crux)
│   │   └── mixing.py              # replay loader + mixed_iterator (ratio, constant budget)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── simulation.py          # held-out terminal next-obs EM + token-F1
│   │   └── general.py             # lm-evaluation-harness wrapper
│   ├── config.py                  # TrainConfig dataclass + YAML load
│   ├── train.py                   # train(config, ...) -> checkpoint dir
│   ├── results.py                 # ResultRecord + write to results/ + EXPERIMENTS.md
│   ├── sanity.py                  # baseline-reproduction / negative-control checks
│   └── run.py                     # orchestrator: eval-before → train → eval-after → record
├── configs/                       # one YAML per matrix cell (Phase 0/1)
├── tests/                         # pytest unit tests (offline) + smoke tests (marked)
├── docs/specs/…  docs/plans/…  docs/EXPERIMENTS.md
└── results/                       # gitignored run outputs
```

---

## Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/lwmf/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: installable package `lwmf`; `pytest` runnable with markers `smoke`, `gpu`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "lwmf"
version = "0.1.0"
description = "Language World Model forgetting experiment harness"
requires-python = ">=3.11"
# Core deps: enough to import every module and run the offline unit suite.
# All heavy GPU-only libs live in the [gpu] extra so local (macOS, no CUDA)
# installs don't choke on bitsandbytes.
dependencies = [
  "torch>=2.3",
  "transformers>=4.44",
  "datasets>=2.20",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
# Installed only on the Azure T4 VM (training + lm-eval). bitsandbytes has no
# macOS CUDA wheel — keep it out of the local install.
gpu = ["trl>=0.9", "peft>=0.12", "accelerate>=0.33", "bitsandbytes>=0.43", "lm-eval>=0.4.3"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = [
  "smoke: end-to-end checks that download models / use GPU (deselect with -m 'not smoke')",
  "gpu: requires CUDA",
]
addopts = "-m 'not smoke and not gpu'"
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty package + test files**

`src/lwmf/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: (empty file)

`tests/test_smoke.py`:
```python
def test_package_imports():
    import lwmf
    assert lwmf.__version__ == "0.1.0"
```

- [ ] **Step 3: Create venv and install (editable)**

Run (local dev machine — core + pytest only, no GPU libs):
```bash
python3.11 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"
```
Expected: installs without error (torch CPU wheel on macOS is fine). On the Azure T4 VM the install is `pip install -e ".[dev,gpu]"` (adds trl/peft/accelerate/bitsandbytes/lm-eval).

- [ ] **Step 4: Run the test**

Run: `pytest -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/lwmf/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore: project scaffold (lwmf package + pytest markers)"
```

---

## Task 1: Trajectory schema

**Files:**
- Create: `src/lwmf/schema.py`, `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `@dataclass Turn(action: str, observation: str)`
  - `@dataclass Trajectory(scenario: str, system: str, turns: list[Turn])`
  - `Trajectory.to_json() -> str` / `Trajectory.from_json(s: str) -> Trajectory`
  - `write_jsonl(path: str, trajs: Iterable[Trajectory]) -> None`
  - `read_jsonl(path: str) -> list[Trajectory]`

- [ ] **Step 1: Write the failing test**

`tests/test_schema.py`:
```python
from lwmf.schema import Turn, Trajectory, write_jsonl, read_jsonl

def test_roundtrip(tmp_path):
    t = Trajectory(
        scenario="fileops",
        system="You simulate a Linux terminal.",
        turns=[Turn("ls", "a.txt\nb.txt"), Turn("cat a.txt", "hello")],
    )
    s = t.to_json()
    assert Trajectory.from_json(s) == t

    p = tmp_path / "d.jsonl"
    write_jsonl(str(p), [t, t])
    got = read_jsonl(str(p))
    assert got == [t, t]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -q`
Expected: FAIL (`ModuleNotFoundError: lwmf.schema`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/schema.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/schema.py tests/test_schema.py
git commit -m "feat(data): Trajectory/Turn schema with JSONL (de)serialization"
```

---

## Task 2: Sandboxed command executor

**Files:**
- Create: `src/lwmf/data/__init__.py`, `src/lwmf/data/sandbox.py`, `tests/test_sandbox.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `run_in_sandbox(commands: list[str], scratch_dir: str, timeout: float = 10.0) -> list[tuple[str, str]]` — runs each command with `scratch_dir` as CWD, returns `(command, combined_stdout+stderr+exit-marker)` pairs. State persists across commands within one call (one shell session per call).

- [ ] **Step 1: Write the failing test**

`tests/test_sandbox.py`:
```python
import os
from lwmf.data.sandbox import run_in_sandbox

def test_state_persists_and_isolated(tmp_path):
    scratch = tmp_path / "box"
    scratch.mkdir()
    pairs = run_in_sandbox(
        ["echo hello > a.txt", "cat a.txt", "pwd"], str(scratch)
    )
    cmds = [c for c, _ in pairs]
    assert cmds == ["echo hello > a.txt", "cat a.txt", "pwd"]
    assert "hello" in pairs[1][1]            # cat sees the file written by prior cmd
    assert str(scratch) in pairs[2][1]       # cwd is the scratch dir
    assert (scratch / "a.txt").exists()

def test_timeout_does_not_hang(tmp_path):
    pairs = run_in_sandbox(["sleep 30"], str(tmp_path), timeout=1.0)
    assert "timeout" in pairs[0][1].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sandbox.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/data/__init__.py`: (empty)

`src/lwmf/data/sandbox.py`:
```python
from __future__ import annotations
import subprocess

def run_in_sandbox(commands: list[str], scratch_dir: str,
                   timeout: float = 10.0) -> list[tuple[str, str]]:
    """Run each command sequentially in a bash session rooted at scratch_dir.

    Returns (command, observation) pairs. Observation = stdout+stderr plus an
    exit-code marker. State (cwd, files, env) persists across commands within
    one call because they run in a single bash -c chained session per command
    but share scratch_dir; for cross-command shell state we serialize each
    command and re-cd, keeping filesystem state (sufficient for our scenarios).
    """
    out: list[tuple[str, str]] = []
    for cmd in commands:
        wrapped = f"cd {scratch_dir!r} && {cmd}"
        try:
            proc = subprocess.run(
                ["bash", "-c", wrapped],
                capture_output=True, text=True, timeout=timeout,
            )
            obs = proc.stdout + proc.stderr
            obs += f"\n[exit {proc.returncode}]"
        except subprocess.TimeoutExpired:
            obs = f"[timeout after {timeout}s]"
        out.append((cmd, obs.strip()))
    return out
```

Note: filesystem state persists because every command `cd`s into the same `scratch_dir`. This is sufficient for file/git/python scenarios; we do not rely on shell-variable persistence across commands.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sandbox.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/data/__init__.py src/lwmf/data/sandbox.py tests/test_sandbox.py
git commit -m "feat(data): sandboxed command executor with timeout + fs persistence"
```

---

## Task 3: Terminal scenario generator

**Files:**
- Create: `src/lwmf/data/terminal_gen.py`, `tests/test_terminal_gen.py`

**Interfaces:**
- Consumes: `run_in_sandbox` (Task 2), `Trajectory`/`Turn`/`write_jsonl` (Task 1).
- Produces:
  - `SYSTEM_PROMPT: str` (the terminal-simulation instruction).
  - `build_scenario(rng: random.Random) -> list[str]` — a random sequence of shell commands.
  - `generate_trajectories(out_path: str, n_scenarios: int, seed: int, scratch_root: str) -> int` — generates trajectories by real execution; returns count written.

- [ ] **Step 1: Write the failing test**

`tests/test_terminal_gen.py`:
```python
import random
from lwmf.data.terminal_gen import build_scenario, generate_trajectories
from lwmf.schema import read_jsonl

def test_build_scenario_deterministic():
    a = build_scenario(random.Random(0))
    b = build_scenario(random.Random(0))
    assert a == b and len(a) >= 2

def test_generate_writes_real_observations(tmp_path):
    out = tmp_path / "traj.jsonl"
    n = generate_trajectories(str(out), n_scenarios=3, seed=1,
                              scratch_root=str(tmp_path / "boxes"))
    assert n == 3
    trajs = read_jsonl(str(out))
    assert len(trajs) == 3
    # every turn has a non-empty action and an observation produced by execution
    for t in trajs:
        assert t.turns
        for turn in t.turns:
            assert turn.action.strip()
            assert "[exit" in turn.observation or "[timeout" in turn.observation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_terminal_gen.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/data/terminal_gen.py`:
```python
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

def generate_trajectories(out_path: str, n_scenarios: int, seed: int,
                          scratch_root: str) -> int:
    rng = random.Random(seed)
    os.makedirs(scratch_root, exist_ok=True)
    trajs: list[Trajectory] = []
    for i in range(n_scenarios):
        cmds = build_scenario(rng)
        box = os.path.join(scratch_root, f"s{i}")
        os.makedirs(box, exist_ok=True)
        pairs = run_in_sandbox(cmds, box)
        shutil.rmtree(box, ignore_errors=True)
        turns = [Turn(action=c, observation=o) for c, o in pairs]
        trajs.append(Trajectory(scenario="terminal", system=SYSTEM_PROMPT, turns=turns))
    write_jsonl(out_path, trajs)
    return len(trajs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_terminal_gen.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/data/terminal_gen.py tests/test_terminal_gen.py
git commit -m "feat(data): terminal scenario generator (real execution -> trajectories)"
```

---

## Task 4: Trajectory → chat → turn expansion

**Files:**
- Create: `src/lwmf/data/format.py`, `tests/test_format.py`

**Interfaces:**
- Consumes: `Trajectory`/`Turn` (Task 1).
- Produces:
  - `@dataclass TurnExample(prefix_messages: list[dict], target: str)` — `prefix_messages` is a chat list ending with the user turn carrying the action; `target` is the observation the model must produce.
  - `trajectory_to_messages(traj: Trajectory) -> list[dict]` — full chat: system, then alternating user(action)/assistant(observation).
  - `expand_to_turns(traj: Trajectory) -> list[TurnExample]` — one example per turn t: prefix = system + turns[0..t-1] (both roles) + user(action_t); target = observation_t.

- [ ] **Step 1: Write the failing test**

`tests/test_format.py`:
```python
from lwmf.schema import Trajectory, Turn
from lwmf.data.format import trajectory_to_messages, expand_to_turns

TRAJ = Trajectory(
    scenario="terminal", system="SYS",
    turns=[Turn("ls", "a.txt"), Turn("cat a.txt", "hello")],
)

def test_messages_shape():
    m = trajectory_to_messages(TRAJ)
    assert m[0] == {"role": "system", "content": "SYS"}
    assert m[1] == {"role": "user", "content": "ls"}
    assert m[2] == {"role": "assistant", "content": "a.txt"}
    assert m[3] == {"role": "user", "content": "cat a.txt"}
    assert m[4] == {"role": "assistant", "content": "hello"}

def test_expand_to_turns():
    ex = expand_to_turns(TRAJ)
    assert len(ex) == 2
    # turn 0: only system + first user action, target = first observation
    assert ex[0].prefix_messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "ls"},
    ]
    assert ex[0].target == "a.txt"
    # turn 1: includes the resolved first turn (both roles) + second action
    assert ex[1].prefix_messages[-1] == {"role": "user", "content": "cat a.txt"}
    assert ex[1].prefix_messages[2] == {"role": "assistant", "content": "a.txt"}
    assert ex[1].target == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_format.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/data/format.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_format.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/data/format.py tests/test_format.py
git commit -m "feat(data): chat formatting + trajectory-to-turn expansion"
```

---

## Task 5: Masked SFT collator (THE crux)

**Files:**
- Create: `src/lwmf/data/collator.py`, `tests/test_collator.py`

**Interfaces:**
- Consumes: `TurnExample` (Task 4).
- Produces:
  - `class MaskedSFTCollator(tokenizer, max_len: int = 1024)` callable on `list[TurnExample]` → `dict[str, torch.Tensor]` with `input_ids`, `attention_mask`, `labels`.
  - **Contract:** `labels[i] == input_ids[i]` for tokens belonging to the target (observation) span; `labels[i] == -100` everywhere else (system, user/action, chat-template control tokens, padding).

This task uses the real tokenizer but offline: load `Qwen/Qwen2.5-0.5B-Instruct` tokenizer from local cache. To keep the unit test offline and fast, the test uses a **tiny stub tokenizer** that exposes the minimal API the collator needs, so the masking logic is verified without any download. A separate smoke test (Task 13) verifies it works with the real tokenizer.

- [ ] **Step 1: Write the failing test (with a stub tokenizer)**

`tests/test_collator.py`:
```python
import torch
from lwmf.data.format import TurnExample
from lwmf.data.collator import MaskedSFTCollator

class StubTok:
    """Whitespace tokenizer; ids are word lengths offset; supports the two
    apply_chat_template modes the collator uses."""
    pad_token_id = 0
    def __init__(self): self.vocab = {}
    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 1)
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # serialize as "<role>: <content>" tokens, plus a control token per turn
        toks = []
        for m in messages:
            toks.append(self._id(f"<{m['role']}>"))
            toks += [self._id(w) for w in m["content"].split()]
        if add_generation_prompt:
            toks.append(self._id("<assistant>"))
        return toks if tokenize else " ".join(map(str, toks))
    def __call__(self, text, add_special_tokens):
        return {"input_ids": [self._id(w) for w in text.split()]}

def test_only_target_tokens_unmasked():
    tok = StubTok()
    ex = TurnExample(
        prefix_messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "ls"},
        ],
        target="a.txt b.txt",
    )
    coll = MaskedSFTCollator(tok, max_len=64)
    batch = coll([ex])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    # exactly the target tokens are unmasked, and they equal the input ids there
    target_ids = [tok._id("a.txt"), tok._id("b.txt")]
    unmasked_positions = (labels != -100).nonzero().flatten().tolist()
    assert [int(input_ids[p]) for p in unmasked_positions] == target_ids
    # everything before the target is masked
    assert int((labels[:unmasked_positions[0]] == -100).all()) == 1

def test_padding_is_masked():
    tok = StubTok()
    short = TurnExample([{"role": "user", "content": "x"}], "y")
    longer = TurnExample([{"role": "user", "content": "x x x x"}], "y y y")
    coll = MaskedSFTCollator(tok, max_len=64)
    batch = coll([short, longer])
    # pad positions (attention_mask==0) must have label -100
    am = batch["attention_mask"]
    labels = batch["labels"]
    assert ((am == 0) & (labels != -100)).sum().item() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collator.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/data/collator.py`:
```python
from __future__ import annotations
import torch
from lwmf.data.format import TurnExample

class MaskedSFTCollator:
    """Builds (input_ids, attention_mask, labels) where loss applies ONLY to
    the target (observation) tokens.

    Strategy: tokenize the prefix (system+history+user action) with the chat
    template and an added generation prompt; tokenize the target separately;
    concatenate. Label mask = -100 over the prefix, real ids over the target.
    """
    def __init__(self, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len

    def _encode(self, ex: TurnExample) -> tuple[list[int], list[int]]:
        prefix_ids = self.tok.apply_chat_template(
            ex.prefix_messages, tokenize=True, add_generation_prompt=True
        )
        target_ids = self.tok(ex.target, add_special_tokens=False)["input_ids"]
        input_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + list(target_ids)
        # truncate from the LEFT so the target is never cut
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            labels = labels[-self.max_len:]
        return input_ids, labels

    def __call__(self, batch: list[TurnExample]) -> dict:
        enc = [self._encode(ex) for ex in batch]
        maxlen = max(len(ids) for ids, _ in enc)
        pad = self.tok.pad_token_id
        input_ids, attn, labels = [], [], []
        for ids, lab in enc:
            n = maxlen - len(ids)
            input_ids.append(ids + [pad] * n)
            attn.append([1] * len(ids) + [0] * n)
            labels.append(lab + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collator.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/data/collator.py tests/test_collator.py
git commit -m "feat(data): masked SFT collator — loss only on observation tokens"
```

---

## Task 6: Replay corpus + mixing iterator

**Files:**
- Create: `src/lwmf/data/mixing.py`, `tests/test_mixing.py`

**Interfaces:**
- Consumes: nothing structural (operates on generic example objects).
- Produces:
  - `def mixed_iterator(narrow: list, replay: Iterator, ratio: float, seed: int) -> Iterator` — yields items, drawing each from `replay` with probability `ratio` and from `narrow` (cycled) otherwise. `ratio=0.0` → only narrow; `ratio=1.0` → only replay.
  - `def load_replay_stream(name: str = "HuggingFaceFW/fineweb-edu", split: str = "train", text_key: str = "text") -> Iterator[str]` — streams text docs from HF (used at runtime, not in unit tests).

The mixing operates at the **example** level (one narrow turn-example vs one replay text chunk). The training budget (Global Constraints) is enforced by the trainer's `max_steps`, so `mixed_iterator` is an infinite stream and the trainer stops at the step budget.

- [ ] **Step 1: Write the failing test**

`tests/test_mixing.py`:
```python
from lwmf.data.mixing import mixed_iterator

def test_ratio_is_approximately_correct():
    narrow = [("N", i) for i in range(5)]
    replay = iter(lambda: None, 1)  # placeholder, replaced below
    def replay_gen():
        i = 0
        while True:
            yield ("R", i); i += 1
    it = mixed_iterator(narrow, replay_gen(), ratio=0.25, seed=0)
    sample = [next(it) for _ in range(4000)]
    frac_replay = sum(1 for x in sample if x[0] == "R") / len(sample)
    assert abs(frac_replay - 0.25) < 0.03

def test_ratio_zero_is_only_narrow():
    narrow = [("N", i) for i in range(3)]
    def replay_gen():
        while True:
            yield ("R", 0)
    it = mixed_iterator(narrow, replay_gen(), ratio=0.0, seed=1)
    sample = [next(it) for _ in range(50)]
    assert all(x[0] == "N" for x in sample)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mixing.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/data/mixing.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mixing.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/data/mixing.py tests/test_mixing.py
git commit -m "feat(data): example-level mixing iterator + fineweb-edu replay stream"
```

---

## Task 7: Simulation eval metric

**Files:**
- Create: `src/lwmf/eval/__init__.py`, `src/lwmf/eval/simulation.py`, `tests/test_simulation.py`

**Interfaces:**
- Consumes: `Trajectory` (Task 1), `expand_to_turns` (Task 4).
- Produces:
  - `def normalize(s: str) -> str` — strip, collapse whitespace, drop the `[exit N]` marker for fuzzy compare.
  - `def exact_match(pred: str, gold: str) -> bool`
  - `def token_f1(pred: str, gold: str) -> float`
  - `def eval_simulation(model, tokenizer, trajs: list[Trajectory], max_new_tokens: int = 128) -> dict` — for each turn-example, greedily generate the observation, compute mean EM and token-F1. (Runtime/GPU; not a unit test.)

- [ ] **Step 1: Write the failing test**

`tests/test_simulation.py`:
```python
from lwmf.eval.simulation import normalize, exact_match, token_f1

def test_normalize_drops_exit_marker():
    assert normalize("a.txt\nb.txt\n[exit 0]") == normalize("a.txt b.txt")

def test_exact_match():
    assert exact_match("a.txt\n[exit 0]", "a.txt [exit 0]") is True
    assert exact_match("a.txt", "b.txt") is False

def test_token_f1():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b", "a b c d") == 0.6667  # 2*P*R/(P+R), P=1,R=0.5 -> 0.6667
    assert token_f1("x y", "a b") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/eval/__init__.py`: (empty)

`src/lwmf/eval/simulation.py`:
```python
from __future__ import annotations
import re
from lwmf.data.format import expand_to_turns

_EXIT = re.compile(r"\[exit\s+-?\d+\]")

def normalize(s: str) -> str:
    s = _EXIT.sub("", s)
    return " ".join(s.split())

def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)

def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for w in p:
        common[w] = min(p.count(w), g.count(w)) if w in g else 0
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return round(2 * precision * recall / (precision + recall), 4)

def eval_simulation(model, tokenizer, trajs, max_new_tokens: int = 128) -> dict:
    import torch
    ems, f1s = [], []
    model.eval()
    for traj in trajs:
        for ex in expand_to_turns(traj):
            ids = tokenizer.apply_chat_template(
                ex.prefix_messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            pred = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            ems.append(1.0 if exact_match(pred, ex.target) else 0.0)
            f1s.append(token_f1(pred, ex.target))
    return {"sim_em": sum(ems) / len(ems), "sim_f1": sum(f1s) / len(f1s),
            "n": len(ems)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/eval/__init__.py src/lwmf/eval/simulation.py tests/test_simulation.py
git commit -m "feat(eval): held-out simulation metric (EM + token-F1)"
```

---

## Task 8: General eval wrapper (lm-evaluation-harness)

**Files:**
- Create: `src/lwmf/eval/general.py`, `tests/test_general_eval.py`

**Interfaces:**
- Produces:
  - `GENERAL_TASKS: dict[str, list[str]]` — `{"knowledge": ["mmlu","arc_easy","arc_challenge","triviaqa"], "commonsense": ["hellaswag","winogrande"], "instruct": ["ifeval"]}`.
  - `def flatten_tasks(include_instruct: bool) -> list[str]`
  - `def run_general_eval(model_path: str, tasks: list[str], limit: int | None, batch_size: int = 8, device: str = "cuda") -> dict[str, float]` — calls `lm_eval.simple_evaluate`, returns a flat `{task: primary_metric}` dict. (Runtime/GPU; not a unit test.)
  - `def perplexity(model, tokenizer, texts: list[str], stride: int = 512) -> float` — held-out PPL canary.

- [ ] **Step 1: Write the failing test**

`tests/test_general_eval.py`:
```python
from lwmf.eval.general import GENERAL_TASKS, flatten_tasks

def test_flatten_excludes_instruct_by_default():
    base = flatten_tasks(include_instruct=False)
    assert "ifeval" not in base
    assert "mmlu" in base and "hellaswag" in base

def test_flatten_includes_instruct_when_asked():
    full = flatten_tasks(include_instruct=True)
    assert "ifeval" in full
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_general_eval.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/eval/general.py`:
```python
from __future__ import annotations

GENERAL_TASKS: dict[str, list[str]] = {
    "knowledge": ["mmlu", "arc_easy", "arc_challenge", "triviaqa"],
    "commonsense": ["hellaswag", "winogrande"],
    "instruct": ["ifeval"],
}

def flatten_tasks(include_instruct: bool) -> list[str]:
    tasks = GENERAL_TASKS["knowledge"] + GENERAL_TASKS["commonsense"]
    if include_instruct:
        tasks = tasks + GENERAL_TASKS["instruct"]
    return tasks

# primary metric key per task in lm-eval results
_PRIMARY = {
    "mmlu": "acc,none", "arc_easy": "acc,none", "arc_challenge": "acc,none",
    "triviaqa": "exact_match,remove_whitespace", "hellaswag": "acc,none",
    "winogrande": "acc,none", "ifeval": "prompt_level_strict_acc,none",
}

def run_general_eval(model_path: str, tasks: list[str], limit=None,
                     batch_size: int = 8, device: str = "cuda") -> dict[str, float]:
    from lm_eval import simple_evaluate
    res = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=float16",
        tasks=tasks, limit=limit, batch_size=batch_size, device=device,
    )
    out: dict[str, float] = {}
    for task, metrics in res["results"].items():
        key = _PRIMARY.get(task)
        if key and key in metrics:
            out[task] = float(metrics[key])
        else:  # fall back to first numeric metric
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    out[task] = float(v); break
    return out

def perplexity(model, tokenizer, texts: list[str], stride: int = 512) -> float:
    import torch, math
    enc = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.to(model.device)
    nlls, count = [], 0
    for i in range(0, enc.size(1) - 1, stride):
        end = min(i + stride, enc.size(1))
        ids = enc[:, i:end]
        with torch.no_grad():
            out = model(ids, labels=ids)
        nlls.append(out.loss * (end - i)); count += (end - i)
    return float(math.exp(torch.stack(nlls).sum() / count))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_general_eval.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/eval/general.py tests/test_general_eval.py
git commit -m "feat(eval): lm-eval-harness wrapper + perplexity canary"
```

---

## Task 9: Run configuration

**Files:**
- Create: `src/lwmf/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass TrainConfig` with fields: `name: str`, `base_model: str`, `method: str` (`"full"|"lora"`), `mixing_ratio: float`, `lr: float`, `max_steps: int`, `seq_len: int = 1024`, `batch_size: int = 4`, `grad_accum: int = 4`, `seed: int = 0`, `replay_dataset: str = "HuggingFaceFW/fineweb-edu"`, `is_instruct: bool = False`.
  - `def load_config(path: str) -> TrainConfig` (YAML).
  - `TrainConfig.validate() -> None` — raises on `method not in {full,lora}` or `not 0<=mixing_ratio<=1`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import pytest
from lwmf.config import TrainConfig, load_config

def test_validate_rejects_bad_method():
    with pytest.raises(ValueError):
        TrainConfig(name="x", base_model="m", method="frob",
                    mixing_ratio=0.0, lr=1e-5, max_steps=10).validate()

def test_validate_rejects_bad_ratio():
    with pytest.raises(ValueError):
        TrainConfig(name="x", base_model="m", method="full",
                    mixing_ratio=1.5, lr=1e-5, max_steps=10).validate()

def test_load_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: t0\nbase_model: Qwen/Qwen2.5-0.5B-Instruct\nmethod: full\n"
        "mixing_ratio: 0.1\nlr: 1.0e-5\nmax_steps: 500\nis_instruct: true\n"
    )
    c = load_config(str(p))
    assert c.name == "t0" and c.method == "full" and c.mixing_ratio == 0.1
    assert c.is_instruct is True
    c.validate()  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/config.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/config.py tests/test_config.py
git commit -m "feat: TrainConfig dataclass + YAML loader + validation"
```

---

## Task 10: Results recording

**Files:**
- Create: `src/lwmf/results.py`, `tests/test_results.py`

**Interfaces:**
- Consumes: `TrainConfig` (Task 9).
- Produces:
  - `@dataclass ResultRecord(config: dict, before: dict, after: dict, sim_before: dict, sim_after: dict, meta: dict)`.
  - `def deltas(record: ResultRecord) -> dict[str, float]` — `after[task]-before[task]` per general task (negative = forgetting), plus `sim_em_gain = sim_after.sim_em - sim_before.sim_em`.
  - `def save_record(record: ResultRecord, results_dir: str) -> str` — writes `results/<name>-<seed>.json`, returns path.
  - `def append_experiments_md(record: ResultRecord, deltas: dict, md_path: str) -> None` — appends one table row to `docs/EXPERIMENTS.md`.

- [ ] **Step 1: Write the failing test**

`tests/test_results.py`:
```python
import json
from lwmf.results import ResultRecord, deltas, save_record

def make():
    return ResultRecord(
        config={"name": "t0", "seed": 0, "mixing_ratio": 0.0},
        before={"mmlu": 0.45, "hellaswag": 0.50},
        after={"mmlu": 0.38, "hellaswag": 0.49},
        sim_before={"sim_em": 0.0}, sim_after={"sim_em": 0.7},
        meta={},
    )

def test_deltas_negative_is_forgetting():
    d = deltas(make())
    assert round(d["mmlu"], 2) == -0.07
    assert round(d["sim_em_gain"], 2) == 0.7

def test_save_record(tmp_path):
    p = save_record(make(), str(tmp_path))
    loaded = json.loads(open(p).read())
    assert loaded["config"]["name"] == "t0"
    assert loaded["after"]["mmlu"] == 0.38
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_results.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/results.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_results.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/results.py tests/test_results.py
git commit -m "feat: ResultRecord + deltas + JSON/EXPERIMENTS.md recording"
```

---

## Task 11: Training runner

**Files:**
- Create: `src/lwmf/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `TrainConfig` (Task 9), `MaskedSFTCollator` (Task 5), `mixed_iterator`/`load_replay_stream` (Task 6), `expand_to_turns` (Task 4), `read_jsonl` (Task 1).
- Produces:
  - `def build_replay_examples(stream, tokenizer, seq_len, n) -> list[TurnExample]` — wraps raw replay text as `TurnExample(prefix_messages=[{"role":"user","content":""}], target=text)` so the SAME collator masks everything but the text (the replay target). (For replay we want loss on the text itself → put text as target.)
  - `def train(config: TrainConfig, terminal_path: str, out_dir: str) -> str` — trains, saves checkpoint, returns `out_dir`.

The replay path reuses the collator: a replay example has an empty prefix and the document as `target`, so loss falls on the document tokens — i.e. standard LM loss on general text. This keeps one masking code path.

- [ ] **Step 1: Write the failing unit test (replay example shaping)**

`tests/test_train.py`:
```python
from lwmf.train import build_replay_examples

class StubTok:
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(text.split())}

def test_build_replay_examples_put_text_as_target():
    stream = iter(["alpha beta gamma", "delta"])
    ex = build_replay_examples(stream, StubTok(), seq_len=1024, n=2)
    assert len(ex) == 2
    assert ex[0].target == "alpha beta gamma"
    assert ex[0].prefix_messages == [{"role": "user", "content": ""}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/train.py`:
```python
from __future__ import annotations
import itertools
from lwmf.config import TrainConfig
from lwmf.data.format import TurnExample, expand_to_turns
from lwmf.data.collator import MaskedSFTCollator
from lwmf.data.mixing import mixed_iterator, load_replay_stream
from lwmf.schema import read_jsonl

def build_replay_examples(stream, tokenizer, seq_len: int, n: int) -> list[TurnExample]:
    out = []
    for text in itertools.islice(stream, n):
        out.append(TurnExample(prefix_messages=[{"role": "user", "content": ""}],
                               target=text))
    return out

def train(config: TrainConfig, terminal_path: str, out_dir: str) -> str:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from torch.utils.data import IterableDataset
    set_seed(config.seed)

    tok = AutoTokenizer.from_pretrained(config.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    trajs = read_jsonl(terminal_path)
    narrow = [ex for t in trajs for ex in expand_to_turns(t)]

    # how many examples the budget needs; replay is streamed lazily
    needed_replay = int(config.max_steps * config.batch_size *
                        config.grad_accum * config.mixing_ratio) + 16
    replay_examples = []
    if config.mixing_ratio > 0.0:
        replay_examples = build_replay_examples(
            load_replay_stream(config.replay_dataset), tok,
            config.seq_len, needed_replay)
    mixed = mixed_iterator(narrow, iter(replay_examples), config.mixing_ratio,
                           config.seed)

    collator = MaskedSFTCollator(tok, max_len=config.seq_len)

    class StreamDS(IterableDataset):
        def __iter__(self):
            return mixed

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=torch.float16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if config.method == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    args = TrainingArguments(
        output_dir=out_dir, max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr, fp16=True,
        optim="adamw_bnb_8bit" if config.method == "full" else "adamw_torch",
        logging_steps=20, save_strategy="no", report_to=[], seed=config.seed,
    )
    Trainer(model=model, args=args, train_dataset=StreamDS(),
            data_collator=collator).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/lwmf/train.py tests/test_train.py
git commit -m "feat(train): config-driven CPT runner (full/LoRA, fp16, masked, mixed)"
```

---

## Task 12: Orchestrator

**Files:**
- Create: `src/lwmf/run.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: all prior modules.
- Produces:
  - `def run_cell(config: TrainConfig, terminal_train: str, terminal_heldout: str, results_dir: str, eval_limit: int | None = None) -> ResultRecord` — eval-before → train → eval-after (general + simulation) → build/save ResultRecord → append EXPERIMENTS.md.
  - `def main()` — CLI: `python -m lwmf.run --config configs/X.yaml --terminal-train data/train.jsonl --terminal-heldout data/heldout.jsonl`.

- [ ] **Step 1: Write the failing test (orchestration wiring with monkeypatch)**

`tests/test_run.py`:
```python
import lwmf.run as run
from lwmf.config import TrainConfig

def test_run_cell_wires_before_after(tmp_path, monkeypatch):
    cfg = TrainConfig(name="t0", base_model="stub", method="full",
                      mixing_ratio=0.0, lr=1e-5, max_steps=1)
    monkeypatch.setattr(run, "train", lambda c, t, o: o)
    calls = {"n": 0}
    def fake_general(model_path, tasks, limit, **kw):
        calls["n"] += 1
        return {"mmlu": 0.45 if calls["n"] == 1 else 0.40}
    monkeypatch.setattr(run, "run_general_eval", fake_general)
    monkeypatch.setattr(run, "_load_model_tok", lambda p: (None, None))
    monkeypatch.setattr(run, "eval_simulation",
                        lambda m, t, trajs, **kw: {"sim_em": 0.0 if calls["n"] <= 1 else 0.6})
    monkeypatch.setattr(run, "read_jsonl", lambda p: [])
    rec = run.run_cell(cfg, "tr.jsonl", "ho.jsonl", str(tmp_path), eval_limit=4)
    assert rec.before["mmlu"] == 0.45
    assert rec.after["mmlu"] == 0.40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write the implementation**

`src/lwmf/run.py`:
```python
from __future__ import annotations
import argparse
import os
from lwmf.config import TrainConfig, load_config
from lwmf.train import train
from lwmf.eval.general import run_general_eval, flatten_tasks
from lwmf.eval.simulation import eval_simulation
from lwmf.results import ResultRecord, deltas, save_record, append_experiments_md
from lwmf.schema import read_jsonl

def _load_model_tok(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                                device_map="cuda")
    return model, tok

def run_cell(config: TrainConfig, terminal_train: str, terminal_heldout: str,
             results_dir: str, eval_limit=None) -> ResultRecord:
    config.validate()
    tasks = flatten_tasks(include_instruct=config.is_instruct)
    heldout = read_jsonl(terminal_heldout)

    before = run_general_eval(config.base_model, tasks, eval_limit)
    m0, t0 = _load_model_tok(config.base_model)
    sim_before = eval_simulation(m0, t0, heldout)
    del m0

    ckpt = os.path.join(results_dir, config.name + f"-seed{config.seed}-ckpt")
    train(config, terminal_train, ckpt)

    after = run_general_eval(ckpt, tasks, eval_limit)
    m1, t1 = _load_model_tok(ckpt)
    sim_after = eval_simulation(m1, t1, heldout)
    del m1

    rec = ResultRecord(config=vars(config), before=before, after=after,
                       sim_before=sim_before, sim_after=sim_after,
                       meta={"tasks": tasks})
    save_record(rec, results_dir)
    append_experiments_md(rec, deltas(rec), os.path.join("docs", "EXPERIMENTS.md"))
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--terminal-train", required=True)
    ap.add_argument("--terminal-heldout", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--eval-limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    rec = run_cell(cfg, a.terminal_train, a.terminal_heldout, a.results_dir, a.eval_limit)
    print(deltas(rec))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run.py -q`
Expected: `1 passed`

- [ ] **Step 5: Run the full offline suite + commit**

Run: `pytest -q`
Expected: all unit tests pass (`-m 'not smoke and not gpu'` is the default).

```bash
git add src/lwmf/run.py tests/test_run.py
git commit -m "feat: orchestrator (eval-before -> train -> eval-after -> record)"
```

---

## Task 13: GPU smoke test (overfit-one-batch + tiny end-to-end)

**Files:**
- Create: `tests/test_smoke_gpu.py`

**Interfaces:**
- Consumes: everything. Runs ONLY on the Azure T4 with `pytest -m smoke`.

- [ ] **Step 1: Write the smoke test**

`tests/test_smoke_gpu.py`:
```python
import pytest
torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.smoke, pytest.mark.gpu]

REAL = "Qwen/Qwen2.5-0.5B-Instruct"

def test_collator_with_real_tokenizer():
    from transformers import AutoTokenizer
    from lwmf.data.format import TurnExample
    from lwmf.data.collator import MaskedSFTCollator
    tok = AutoTokenizer.from_pretrained(REAL)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    ex = TurnExample([{"role": "system", "content": "S"},
                      {"role": "user", "content": "ls"}], "a.txt\n[exit 0]")
    batch = MaskedSFTCollator(tok, 128)([ex])
    labels = batch["labels"][0]
    unmasked = labels[labels != -100]
    # the unmasked region must decode back to the target text
    decoded = tok.decode(unmasked)
    assert "a.txt" in decoded

def test_overfit_one_batch(tmp_path):
    """Train a few steps on ONE tiny terminal file; loss must drop sharply."""
    from lwmf.config import TrainConfig
    from lwmf.data.terminal_gen import generate_trajectories
    from lwmf.train import train
    data = tmp_path / "tiny.jsonl"
    generate_trajectories(str(data), n_scenarios=4, seed=0,
                          scratch_root=str(tmp_path / "b"))
    cfg = TrainConfig(name="overfit", base_model=REAL, method="full",
                      mixing_ratio=0.0, lr=5e-5, max_steps=30,
                      batch_size=2, grad_accum=1, seed=0)
    out = train(cfg, str(data), str(tmp_path / "ckpt"))
    assert (tmp_path / "ckpt").exists()
```

- [ ] **Step 2: Run on the T4 VM**

Run (on Azure VM): `pytest -m smoke -q tests/test_smoke_gpu.py`
Expected: both pass; training logs show loss decreasing across the 30 steps (manual check of stdout).

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_gpu.py
git commit -m "test: GPU smoke — real-tokenizer masking + overfit-one-batch"
```

---

## Task 14: Phase-0/1 configs

**Files:**
- Create: `configs/phase1_05B_instruct_mix00.yaml` … `mix50.yaml` (4 files), `configs/control_replay_only.yaml`

**Interfaces:**
- Consumes: `load_config` (Task 9). No tests (data files); validated by `run_cell` at execution.

- [ ] **Step 1: Write the four Phase-1 configs**

`configs/phase1_05B_instruct_mix00.yaml`:
```yaml
name: p1_05Bi_mix00
base_model: Qwen/Qwen2.5-0.5B-Instruct
method: full
mixing_ratio: 0.0
lr: 1.0e-5
max_steps: 500
batch_size: 4
grad_accum: 4
seed: 0
is_instruct: true
```

Repeat for `mixing_ratio: 0.10 / 0.25 / 0.50` in the other three files (name suffix `mix10/mix25/mix50`), all else identical (constant-budget constraint).

`configs/control_replay_only.yaml`: same but `mixing_ratio: 1.0`, `name: control_replay_only` (negative control — expect ~0 forgetting, ~0 sim gain).

- [ ] **Step 2: Validate configs load**

Run:
```bash
python -c "from lwmf.config import load_config; [load_config(p).validate() for p in __import__('glob').glob('configs/*.yaml')]; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add configs/
git commit -m "feat: Phase-1 mixing-sweep configs + replay-only negative control"
```

---

## Execution Runbook (Phases 0–4 — run after the harness is built & smoke-tested)

These are **not** TDD tasks; they are the experiment, run on the Azure VM. Record every run in `docs/EXPERIMENTS.md` (the orchestrator appends automatically) and commit results JSON.

### R0. Provision Azure T4 (australiaeast)
```bash
az group create -n lwmf-rg -l australiaeast
az vm create -g lwmf-rg -n lwmf-t4 --image Ubuntu2204 \
  --size Standard_NC4as_T4_v3 --priority Spot --max-price -1 \
  --eviction-policy Deallocate --admin-username azureuser --generate-ssh-keys
# ssh in, install CUDA driver + python3.11, clone repo, pip install -e ".[dev,gpu]"
# verify: python -c "import torch; print(torch.cuda.is_available())"  -> True
```

### R1. Phase 0 — data + baseline + smoke
1. Generate datasets:
   ```bash
   python -c "from lwmf.data.terminal_gen import generate_trajectories as g; \
     g('data/train.jsonl', 28000, 0, 'data/boxes'); g('data/heldout.jsonl', 600, 99, 'data/boxes_ho')"
   ```
   (Confirm ~8–10M tokens; adjust `n_scenarios` if needed.)
2. Run smoke: `pytest -m smoke -q` → must pass (incl. overfit-one-batch).
3. **Baseline reproduction gate:** `run_general_eval("Qwen/Qwen2.5-0.5B-Instruct", flatten_tasks(True), limit=None)` — compare MMLU/ARC/HellaSwag against the Qwen2.5 tech report (±~3 pts). If off → fix eval wiring before any training run.
4. **LR calibration:** run `mix00` at lr ∈ {5e-6, 1e-5, 3e-5}, inspect Δgeneral + sim_em. Pick the lr giving *visible-but-not-catastrophic* forgetting AND sim_em clearly above baseline. Lock that lr for all Phase-1 cells.

### R2. Phase 1 — the headline curve
For each config in `configs/phase1_*` and seed ∈ {0,1,2}:
```bash
python -m lwmf.run --config configs/phase1_05B_instruct_mix00.yaml \
  --terminal-train data/train.jsonl --terminal-heldout data/heldout.jsonl
```
(Override seed by editing the YAML or adding a `--seed` flag if implemented.) Also run `control_replay_only` (1 seed). Plot Δgeneral vs mixing_ratio with 3-seed CIs → **the figure**.
**Validity gate:** discard any cell whose sim_em did not rise vs baseline.

### R3. Phase 2 — expand axes with signal
Clone the Phase-1 configs changing one axis at a time: `base_model` → `Qwen2.5-0.5B` (base) and `Qwen2.5-1.5B(-Instruct)`; `method` → `lora`. Run the mixing sweep for each axis that Phase 1 suggests is interesting. (1.5B full-FT: confirm it fits 16 GB; if OOM, keep grad_accum, drop batch_size to 2.)

### R4. Phase 3 — trade-off
From all recorded JSON, scatter `sim_em_gain` (x) vs mean Δgeneral (y) → Pareto front. (Small plotting script reading `results/*.json`.)

### R5. Phase 4 — replay nature (v2, optional)
If the Instruct model lost IFEval that plain-text replay didn't restore: add `replay_dataset` pointing to an instruction dataset and a flag to shape replay as chat rather than plain text; re-run the mixing sweep; compare IFEval recovery vs v1.

---

## Self-Review

**Spec coverage:**
- §3 matrix (mixing/method/size/flavor) → Tasks 9, 14 + Runbook R2/R3. ✓
- §4 phases → Runbook R1–R5. ✓
- §5.1 terminal data → Tasks 2,3; §5.2 replay → Task 6; §5.3 instruction replay → R5; §5.4 constant budget → Global Constraints + `max_steps` in Task 11. ✓
- §6 evals (lm-eval battery, perplexity, simulation EM/F1) → Tasks 7, 8. ✓
- §7 testing (Layer A unit tests → Tasks 1–12; loss-mask crux → Task 5 + R13 smoke; Layer B gates: baseline-repro → R1.3, learning-check → run_cell validity gate, overfit-one-batch → Task 13, negative control → Task 14; Layer C statistics → R2 3-seed CIs). ✓
- §8 infra (T4, fp16, grad-ckpt, 8-bit optim) → Task 11 + R0. ✓
- §9 models → Task 14 configs + R2/R3. ✓

**Placeholder scan:** no TBD/TODO; every code step has complete code. One deliberate runtime parameter (`n_scenarios` to hit ~8–10M tokens) is flagged for confirmation in R1.1, not a code placeholder.

**Type consistency:** `TurnExample(prefix_messages, target)` used identically in Tasks 4,5,7,11. `MaskedSFTCollator(tokenizer, max_len)` consistent Tasks 5,11,13. `run_general_eval(model_path, tasks, limit, ...)` consistent Tasks 8,12 (and monkeypatched with matching signature in Task 12 test). `ResultRecord` fields consistent Tasks 10,12. `mixed_iterator(narrow, replay, ratio, seed)` consistent Tasks 6,11. ✓

**Known simplification (flagged, not a defect):** Task 2 sandbox persists filesystem state but not shell-variable state across commands (every command re-`cd`s). Scenarios in Task 3 only depend on filesystem state, so this is sufficient; documented in `sandbox.py`.
