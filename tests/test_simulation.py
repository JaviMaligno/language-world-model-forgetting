from lwmf.eval.simulation import normalize, exact_match, token_f1

def test_normalize_drops_exit_marker():
    assert normalize("a.txt\nb.txt\n[exit 0]") == normalize("a.txt b.txt")

def test_exact_match():
    assert exact_match("a.txt\n[exit 0]", "a.txt [exit 0]") is True
    assert exact_match("a.txt", "b.txt") is False

def test_token_f1():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b", "a b c d") == 0.6667  # 2*P*R/(P+R), P=1,R=0.5 -> 0.6667
    assert token_f1("x y", "a b") == 0.0
