from __future__ import annotations
import torch
from lwmf.data.format import TurnExample

class MaskedSFTCollator:
    """Builds (input_ids, attention_mask, labels) where loss applies ONLY to
    the target (observation) tokens.

    Strategy: tokenize the prefix (system+history+user action) with the chat
    template and an added generation prompt; tokenize the target separately;
    concatenate. Label mask = -100 over the prefix, real ids over the target.
    """
    def __init__(self, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len

    def _encode(self, ex: TurnExample) -> tuple[list[int], list[int]]:
        prefix_ids = self.tok.apply_chat_template(
            ex.prefix_messages, tokenize=True, add_generation_prompt=True
        )
        target_ids = self.tok(ex.target, add_special_tokens=False)["input_ids"]
        input_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + list(target_ids)
        # truncate from the LEFT so the target is never cut
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            labels = labels[-self.max_len:]
        return input_ids, labels

    def __call__(self, batch: list[TurnExample]) -> dict:
        enc = [self._encode(ex) for ex in batch]
        maxlen = max(len(ids) for ids, _ in enc)
        pad = self.tok.pad_token_id
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
