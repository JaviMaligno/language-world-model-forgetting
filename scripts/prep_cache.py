#!/usr/bin/env python
"""Populate the local HF cache (ONLINE, with retries) with the models + eval
datasets that the offline training jobs need, so the cache can be tarred and
reused with HF_HUB_OFFLINE=1. Run before taring $HOME/.cache/huggingface.
"""
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from lwmf.eval.general import run_general_eval, flatten_tasks

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def once():
    for m in MODELS:
        AutoTokenizer.from_pretrained(m)
        AutoModelForCausalLM.from_pretrained(m)
    # downloads arc/hellaswag/winogrande into the cache
    run_general_eval("Qwen/Qwen2.5-0.5B-Instruct", flatten_tasks(True), limit=1)


for attempt in range(10):
    try:
        once()
        print("CACHE OK (3 models + datasets)")
        break
    except Exception as e:  # noqa: BLE001
        print(f"retry {attempt}: {str(e)[:140]}")
        time.sleep(15)
else:
    raise SystemExit("cache prep failed after retries")
