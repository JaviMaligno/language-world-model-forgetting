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
            # Skip no-output turns (e.g. mkdir/echo>/rm): their observation is just
            # the exit marker, which normalize() strips to "" -> trivially matched and
            # inflating/saturating sim_em. Score only turns with real stdout, the
            # informative test of world-model fidelity.
            if not normalize(ex.target):
                continue
            ids = tokenizer.apply_chat_template(
                ex.prefix_messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=False,
            )
            # Some transformers versions return a BatchEncoding/dict even with
            # return_dict=False; normalize to the input_ids tensor.
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            ids = ids.to(model.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            pred = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            ems.append(1.0 if exact_match(pred, ex.target) else 0.0)
            f1s.append(token_f1(pred, ex.target))
    if not ems:
        return {"sim_em": 0.0, "sim_f1": 0.0, "n": 0}
    return {"sim_em": sum(ems) / len(ems), "sim_f1": sum(f1s) / len(f1s),
            "n": len(ems)}
