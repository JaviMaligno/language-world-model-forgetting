from lwmf.schema import Turn, Trajectory, write_jsonl, read_jsonl

def test_roundtrip(tmp_path):
    t = Trajectory(
        scenario="fileops",
        system="You simulate a Linux terminal.",
        turns=[Turn("ls", "a.txt\nb.txt"), Turn("cat a.txt", "hello")],
    )
    s = t.to_json()
    assert Trajectory.from_json(s) == t

    p = tmp_path / "d.jsonl"
    write_jsonl(str(p), [t, t])
    got = read_jsonl(str(p))
    assert got == [t, t]
