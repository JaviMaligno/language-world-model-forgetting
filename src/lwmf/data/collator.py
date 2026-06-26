from __future__ import annotations
import warnings
import torch
from lwmf.data.format import TurnExample

class MaskedSFTCollator:
    """Builds (input_ids, attention_mask, labels) where loss applies ONLY to
    the target (observation) tokens plus a trailing EOS (so the model learns to
    stop). Prefix (system + history + user action), chat-template control
    tokens, and padding are all masked to -100.
    """
    def __init__(self, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len

    def _prefix_ids(self, ex: TurnExample) -> list[int]:
        enc = self.tok.apply_chat_template(
            ex.prefix_messages, tokenize=True, add_generation_prompt=True,
            return_dict=False,
        )
        # Some transformers versions return a BatchEncoding/dict even with
        # return_dict=False; normalize to a flat list of token ids.
        if isinstance(enc, dict):
            enc = enc["input_ids"]
        return list(enc)

    def _encode(self, ex: TurnExample) -> tuple[list[int], list[int]]:
        prefix_ids = self._prefix_ids(ex)
        target_ids = list(self.tok(ex.target, add_special_tokens=False)["input_ids"])
        eos = self.tok.eos_token_id
        if eos is not None:
            target_ids = target_ids + [eos]
        input_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + target_ids
        # truncate from the LEFT so the target (at the end) is preserved;
        # if the target itself exceeds max_len, its leading tokens are dropped.
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            labels = labels[-self.max_len:]
        if all(l == -100 for l in labels):
            warnings.warn(
                "MaskedSFTCollator: example has no supervised (unmasked) tokens "
                "after truncation; it contributes zero gradient."
            )
        return input_ids, labels

    def __call__(self, batch: list[TurnExample]) -> dict:
        if not batch:
            raise ValueError("MaskedSFTCollator received an empty batch")
        enc = [self._encode(ex) for ex in batch]
        maxlen = max(len(ids) for ids, _ in enc)
        pad = self.tok.pad_token_id
        if pad is None:
            pad = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0
        input_ids, attn, labels = [], [], []
        for ids, lab in enc:
            n = maxlen - len(ids)
            input_ids.append(ids + [pad] * n)
            attn.append([1] * len(ids) + [0] * n)
            labels.append(lab + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
