from __future__ import annotations
import argparse
import os
from lwmf.config import TrainConfig, load_config
from lwmf.train import train
from lwmf.eval.general import run_general_eval, flatten_tasks
from lwmf.eval.simulation import eval_simulation
from lwmf.results import ResultRecord, deltas, save_record, append_experiments_md
from lwmf.schema import read_jsonl

def _free():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    del m0, t0
    _free()

    ckpt = os.path.join(results_dir, config.name + f"-seed{config.seed}-ckpt")
    train(config, terminal_train, ckpt)

    after = run_general_eval(ckpt, tasks, eval_limit)
    m1, t1 = _load_model_tok(ckpt)
    sim_after = eval_simulation(m1, t1, heldout)
    del m1, t1
    _free()

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
