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
