#!/usr/bin/env python
"""Prep for Batch A (ONLINE, with retries):
  1. Restore the existing HF cache tgz, then download the IFEval dataset into it
     (running lm-eval ifeval at limit=1) and re-tar -> new cache with ifeval.
  2. Generate a HARD held-out terminal set (OOD commands: sed/awk/grep/pipes).

Usage: python scripts/prep_batcha.py <cache_in_tgz> <cache_out_dir> <hard_out_dir>
"""
import os
import sys
import time
import tarfile

cache_in = sys.argv[1]      # path to existing hfcache.tgz
cache_out = sys.argv[2]     # output dir for new hfcache.tgz
hard_out = sys.argv[3]      # output dir for hard heldout jsonl

# 1) restore existing cache into $HOME/.cache/huggingface
home = os.path.expanduser("~")
with tarfile.open(cache_in) as t:
    t.extractall(home)
print("restored cache", flush=True)

# 2) download ifeval (+ deps already pip-installed by caller) via lm-eval at limit=1
from lm_eval import simple_evaluate  # noqa: E402
for attempt in range(10):
    try:
        simple_evaluate(model="hf",
                        model_args="pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=float16",
                        tasks=["ifeval"], limit=1, batch_size=1, device="cuda")
        print("IFEVAL DATASET CACHED", flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print(f"ifeval retry {attempt}: {str(e)[:120]}", flush=True)
        time.sleep(15)
else:
    raise SystemExit("ifeval cache failed")

# 3) re-tar the (now ifeval-augmented) cache
os.makedirs(cache_out, exist_ok=True)
with tarfile.open(os.path.join(cache_out, "hfcache.tgz"), "w:gz") as t:
    t.add(os.path.join(home, ".cache/huggingface"), arcname=".cache/huggingface")
print("re-tarred cache with ifeval", flush=True)

# 4) generate hard held-out terminal set (OOD commands)
from lwmf.data.terminal_gen import generate_trajectories, build_scenario_hard  # noqa: E402
os.makedirs(hard_out, exist_ok=True)
n = generate_trajectories(os.path.join(hard_out, "heldout_hard.jsonl"),
                          n_scenarios=400, seed=123, scratch_root="/tmp/boxes_hard",
                          scenario_fn=build_scenario_hard)
print(f"HARD HELDOUT: {n} scenarios", flush=True)
