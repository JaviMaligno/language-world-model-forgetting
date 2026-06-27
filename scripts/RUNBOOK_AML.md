# Azure ML run guide (compute instance `lwmf-t4`)

This experiment runs on an **Azure ML compute instance** (not a raw VM) because the
only GPU-capable scope available is the AML workspace `model_server` (australiaeast),
which has the T4 quota. The compute instance `lwmf-t4` is a `Standard_NC4as_T4_v3`
(4 vCPU, 1× T4 16 GB) with NVIDIA driver + CUDA preinstalled.

## Cost control (IMPORTANT)
- The instance **bills ~$0.53/hr while Running**. It has `idle_time_before_shutdown=30min`.
- **Stop it when not in use.** Disk/state persist across stop/start, so the venv survives.
- Start/stop options:
  - AML Studio: Compute → `lwmf-t4` → Stop / Start.
  - SDK (from the local `aml-venv`):
    ```python
    from azure.identity import AzureCliCredential
    from azure.ai.ml import MLClient
    ml = MLClient(AzureCliCredential(), "26322bb9-38d2-477e-9c5f-a048c8a023d7",
                  "datascience", "model_server")
    ml.compute.begin_stop("lwmf-t4").result()   # or begin_start
    ```

## Access
SSH public access is disabled. Operate via **AML Studio → Compute → lwmf-t4 → Terminal**
(or Jupyter). All commands below run in that web terminal.

## One-time bootstrap
```bash
# authenticate git for the private repo first (gh auth login, or a PAT), then:
curl -fsSL https://raw.githubusercontent.com/JaviMaligno/language-world-model-forgetting/master/scripts/compute_bootstrap.sh -o /tmp/bootstrap.sh
bash /tmp/bootstrap.sh
# (or: git clone the repo and run bash scripts/compute_bootstrap.sh)
cd ~/language-world-model-forgetting && source .venv/bin/activate
```

## Phase 0 — data + baseline + smoke
```bash
# 1) Generate terminal trajectories (~8-10M tokens train + held-out)
python -c "from lwmf.data.terminal_gen import generate_trajectories as g; \
  g('data/train.jsonl', 28000, 0, 'data/boxes'); g('data/heldout.jsonl', 600, 99, 'data/boxes_ho')"

# 2) GPU smoke (overfit-one-batch + real-tokenizer mask)
pytest -q -m smoke

# 3) Baseline reproduction gate — compare to the Qwen2.5 tech report (±~3 pts)
python -c "from lwmf.eval.general import run_general_eval, flatten_tasks; \
  print(run_general_eval('Qwen/Qwen2.5-0.5B-Instruct', flatten_tasks(True), limit=200))"

# 4) LR calibration: run mix00 at a few LRs, pick visible-but-not-catastrophic forgetting
python -m lwmf.run --config configs/phase1_05B_instruct_mix00.yaml \
  --terminal-train data/train.jsonl --terminal-heldout data/heldout.jsonl --eval-limit 200
```

## Phase 1 — the headline curve
```bash
for c in mix00 mix10 mix25 mix50; do
  python -m lwmf.run --config configs/phase1_05B_instruct_${c}.yaml \
    --terminal-train data/train.jsonl --terminal-heldout data/heldout.jsonl
done
python -m lwmf.run --config configs/control_replay_only.yaml \
  --terminal-train data/train.jsonl --terminal-heldout data/heldout.jsonl
# results land in results/*.json and docs/EXPERIMENTS.md ; commit/push them.
```

Validity gate: discard any cell whose held-out sim_em did not rise vs baseline.
See `docs/plans/2026-06-26-lwm-forgetting-implementation.md` (Runbook R2–R5) for Phases 2–4.
