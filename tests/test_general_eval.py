from lwmf.eval.general import GENERAL_TASKS, flatten_tasks

def test_flatten_excludes_instruct_by_default():
    base = flatten_tasks(include_instruct=False)
    assert "ifeval" not in base
    assert "mmlu" in base and "hellaswag" in base

def test_flatten_includes_instruct_when_asked():
    full = flatten_tasks(include_instruct=True)
    assert "ifeval" in full
