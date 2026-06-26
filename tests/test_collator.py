import torch
from lwmf.data.format import TurnExample
from lwmf.data.collator import MaskedSFTCollator

class StubTok:
    """Whitespace tokenizer; ids are word lengths offset; supports the two
    apply_chat_template modes the collator uses."""
    pad_token_id = 0
    def __init__(self): self.vocab = {}
    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 1)
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # serialize as "<role>: <content>" tokens, plus a control token per turn
        toks = []
        for m in messages:
            toks.append(self._id(f"<{m['role']}>"))
            toks += [self._id(w) for w in m["content"].split()]
        if add_generation_prompt:
            toks.append(self._id("<assistant>"))
        return toks if tokenize else " ".join(map(str, toks))
    def __call__(self, text, add_special_tokens):
        return {"input_ids": [self._id(w) for w in text.split()]}

def test_only_target_tokens_unmasked():
    tok = StubTok()
    ex = TurnExample(
        prefix_messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "ls"},
        ],
        target="a.txt b.txt",
    )
    coll = MaskedSFTCollator(tok, max_len=64)
    batch = coll([ex])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    # exactly the target tokens are unmasked, and they equal the input ids there
    target_ids = [tok._id("a.txt"), tok._id("b.txt")]
    unmasked_positions = (labels != -100).nonzero().flatten().tolist()
    assert [int(input_ids[p]) for p in unmasked_positions] == target_ids
    # everything before the target is masked
    assert int((labels[:unmasked_positions[0]] == -100).all()) == 1

def test_padding_is_masked():
    tok = StubTok()
    short = TurnExample([{"role": "user", "content": "x"}], "y")
    longer = TurnExample([{"role": "user", "content": "x x x x"}], "y y y")
    coll = MaskedSFTCollator(tok, max_len=64)
    batch = coll([short, longer])
    # pad positions (attention_mask==0) must have label -100
    am = batch["attention_mask"]
    labels = batch["labels"]
    assert ((am == 0) & (labels != -100)).sum().item() == 0
