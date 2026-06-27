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
