from lwmf.data.mixing import mixed_iterator

def test_ratio_is_approximately_correct():
    narrow = [("N", i) for i in range(5)]
    def replay_gen():
        i = 0
        while True:
            yield ("R", i); i += 1
    it = mixed_iterator(narrow, replay_gen(), ratio=0.25, seed=0)
    sample = [next(it) for _ in range(4000)]
    frac_replay = sum(1 for x in sample if x[0] == "R") / len(sample)
    assert abs(frac_replay - 0.25) < 0.03

def test_ratio_zero_is_only_narrow():
    narrow = [("N", i) for i in range(3)]
    def replay_gen():
        while True:
            yield ("R", 0)
    it = mixed_iterator(narrow, replay_gen(), ratio=0.0, seed=1)
    sample = [next(it) for _ in range(50)]
    assert all(x[0] == "N" for x in sample)
