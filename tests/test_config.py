import pytest
from lwmf.config import TrainConfig, load_config

def test_validate_rejects_bad_method():
    with pytest.raises(ValueError):
        TrainConfig(name="x", base_model="m", method="frob",
                    mixing_ratio=0.0, lr=1e-5, max_steps=10).validate()

def test_validate_rejects_bad_ratio():
    with pytest.raises(ValueError):
        TrainConfig(name="x", base_model="m", method="full",
                    mixing_ratio=1.5, lr=1e-5, max_steps=10).validate()

def test_load_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: t0\nbase_model: Qwen/Qwen2.5-0.5B-Instruct\nmethod: full\n"
        "mixing_ratio: 0.1\nlr: 1.0e-5\nmax_steps: 500\nis_instruct: true\n"
    )
    c = load_config(str(p))
    assert c.name == "t0" and c.method == "full" and c.mixing_ratio == 0.1
    assert c.is_instruct is True
    c.validate()  # should not raise
