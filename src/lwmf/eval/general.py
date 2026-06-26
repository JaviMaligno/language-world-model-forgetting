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
