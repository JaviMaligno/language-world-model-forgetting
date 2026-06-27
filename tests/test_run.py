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
    monkeypatch.setattr(run, "append_experiments_md", lambda *a, **k: None)
    rec = run.run_cell(cfg, "tr.jsonl", "ho.jsonl", str(tmp_path), eval_limit=4)
    assert rec.before["mmlu"] == 0.45
    assert rec.after["mmlu"] == 0.40
