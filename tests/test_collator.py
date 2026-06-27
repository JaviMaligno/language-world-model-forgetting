import pytest
import torch
from lwmf.data.format import TurnExample
from lwmf.data.collator import MaskedSFTCollator


class StubTok:
    """Whitespace tokenizer; ids are word lengths offset; supports the two
    apply_chat_template modes the collator uses.

    Updated to match real transformers call signature:
    - eos_token_id attribute added
    - apply_chat_template accepts return_dict keyword argument
    - when tokenize=True and return_dict=False, returns plain list of ids
    """
    pad_token_id = 0
    eos_token_id = 999

    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 1)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            return_dict=False):
        # serialize as "<role>: <content>" tokens, plus a control token per turn
        toks = []
        for m in messages:
            toks.append(self._id(f"<{m['role']}>"))
            toks += [self._id(w) for w in m["content"].split()]
        if add_generation_prompt:
            toks.append(self._id("<assistant>"))
        if not tokenize:
            return " ".join(map(str, toks))
        # return_dict=False (or any value) => return plain list, consistent with
        # what the collator expects after normalisation
        return toks

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

    # Target ids + EOS are the unmasked tokens
    target_word_ids = [tok._id("a.txt"), tok._id("b.txt")]
    eos = tok.eos_token_id
    expected_unmasked_ids = target_word_ids + [eos]

    unmasked_positions = (labels != -100).nonzero().flatten().tolist()

    # (a) unmasked input_ids equal target_ids + eos
    assert [int(input_ids[p]) for p in unmasked_positions] == expected_unmasked_ids

    # (b) mask-value contract: labels at unmasked positions equal input_ids there
    for p in unmasked_positions:
        assert int(labels[p]) == int(input_ids[p]), (
            f"labels[{p}]={int(labels[p])} != input_ids[{p}]={int(input_ids[p])}"
        )

    # everything before the target (prefix) is masked
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


def test_truncation_preserves_target():
    """Build a TurnExample whose prefix tokenizes to more than max_len tokens.
    With max_len=8, a long prefix forces left-truncation, so the target tokens
    at the end must still appear in the output.
    """
    tok = StubTok()
    # prefix content: 6 distinct words -> prefix_ids will be:
    #   <system> w1 w2 w3 w4 w5 w6 <assistant>  => 8 tokens
    # target: "tgt" + EOS => 2 tokens => total 10 > max_len=8
    # After left-truncation to 8: last 8 tokens = w5 w6 <assistant> tgt EOS + 3 more
    # The point is: target tokens (tgt + EOS) survive at the right end.
    ex = TurnExample(
        prefix_messages=[
            {"role": "system", "content": "w1 w2 w3 w4 w5 w6"},
        ],
        target="tgt",
    )
    max_len = 8
    coll = MaskedSFTCollator(tok, max_len=max_len)
    batch = coll([ex])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    # Shape invariants
    assert len(input_ids) == max_len
    assert len(labels) == max_len

    # Target must be present: tgt and EOS appear at the end
    tgt_id = tok._id("tgt")
    eos_id = tok.eos_token_id
    assert int(input_ids[-1]) == eos_id, "EOS must be at last position"
    assert int(input_ids[-2]) == tgt_id, "target token must precede EOS"

    # labels and input_ids have the same length
    assert input_ids.shape == labels.shape


@pytest.mark.smoke
def test_collator_real_tokenizer_mask():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ex = TurnExample(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "ls"}],
        "a.txt\n[exit 0]",
    )
    batch = MaskedSFTCollator(tok, 128)([ex])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    unmasked = (labels != -100).nonzero().flatten()
    assert len(unmasked) > 0
    # mask-value contract on the real tokenizer
    assert (labels[unmasked] == input_ids[unmasked]).all()
    decoded = tok.decode(input_ids[unmasked].tolist())
    assert "a.txt" in decoded
