from lwmf.data.sandbox import run_in_sandbox

def test_state_persists_and_isolated(tmp_path):
    scratch = tmp_path / "box"
    scratch.mkdir()
    pairs = run_in_sandbox(
        ["echo hello > a.txt", "cat a.txt", "pwd"], str(scratch)
    )
    cmds = [c for c, _ in pairs]
    assert cmds == ["echo hello > a.txt", "cat a.txt", "pwd"]
    assert "hello" in pairs[1][1]            # cat sees the file written by prior cmd
    assert str(scratch) in pairs[2][1]       # cwd is the scratch dir
    assert (scratch / "a.txt").exists()

def test_timeout_does_not_hang(tmp_path):
    pairs = run_in_sandbox(["sleep 30"], str(tmp_path), timeout=1.0)
    assert "timeout" in pairs[0][1].lower()
