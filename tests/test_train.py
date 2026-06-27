from lwmf.train import build_replay_examples

class StubTok:
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(text.split())}

def test_build_replay_examples_put_text_as_target():
    stream = iter(["alpha beta gamma", "delta"])
    ex = build_replay_examples(stream, StubTok(), seq_len=1024, n=2)
    assert len(ex) == 2
    assert ex[0].target == "alpha beta gamma"
    assert ex[0].prefix_messages == [{"role": "user", "content": ""}]
