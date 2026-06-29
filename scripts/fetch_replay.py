#!/usr/bin/env python
"""Stream N docs of a replay corpus (default FineWeb-Edu) to a JSONL file, with
retries. Run ONLINE (no HF_HUB_OFFLINE). Used to pre-stage replay text as a data
asset so training jobs can read it offline via LWMF_REPLAY_FILE.

Usage: python scripts/fetch_replay.py <out.jsonl> [n_docs] [dataset] [config]

Default source is English Wikipedia (parquet, uniform schema, streams cleanly).
FineWeb-Edu was the original intent but its shards have inconsistent Arrow schemas
that raise "Couldn't cast" under streaming; Wikipedia is an equally valid broad
general-text corpus for anti-forgetting replay.
"""
import json
import sys
import time
from datasets import load_dataset

out_path = sys.argv[1]
n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
dataset = sys.argv[3] if len(sys.argv) > 3 else "wikimedia/wikipedia"
config = sys.argv[4] if len(sys.argv) > 4 else "20231101.en"


def grab():
    if config:
        ds = load_dataset(dataset, config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset, split="train", streaming=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as o:
        for row in ds:
            t = row.get("text")
            if t:
                o.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                n += 1
            if n >= n_docs:
                break
    return n


for attempt in range(10):
    try:
        n = grab()
        print(f"WROTE {n} docs to {out_path}")
        break
    except Exception as e:  # noqa: BLE001
        print(f"retry {attempt}: {str(e)[:140]}")
        time.sleep(15)
else:
    raise SystemExit("replay prefetch failed after retries")
