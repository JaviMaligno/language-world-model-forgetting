from __future__ import annotations
import itertools
from lwmf.config import TrainConfig
from lwmf.data.format import TurnExample, expand_to_turns
from lwmf.data.collator import MaskedSFTCollator
from lwmf.data.mixing import mixed_iterator, load_replay_stream
from lwmf.schema import read_jsonl


def build_replay_examples(stream, tokenizer, seq_len: int, n: int) -> list[TurnExample]:
    out = []
    for text in itertools.islice(stream, n):
        out.append(TurnExample(prefix_messages=[{"role": "user", "content": ""}],
                               target=text))
    return out


def train(config: TrainConfig, terminal_path: str, out_dir: str) -> str:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from torch.utils.data import IterableDataset
    set_seed(config.seed)

    tok = AutoTokenizer.from_pretrained(config.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    trajs = read_jsonl(terminal_path)
    narrow = [ex for t in trajs for ex in expand_to_turns(t)]

    # how many examples the budget needs; replay is streamed lazily
    needed_replay = int(config.max_steps * config.batch_size *
                        config.grad_accum * config.mixing_ratio) + 16
    replay_examples = []
    if config.mixing_ratio > 0.0:
        replay_examples = build_replay_examples(
            load_replay_stream(config.replay_dataset), tok,
            config.seq_len, needed_replay)

    def _make_mixed():
        replay_iter = (
            itertools.cycle(replay_examples) if replay_examples else iter(())
        )
        return mixed_iterator(narrow, replay_iter, config.mixing_ratio, config.seed)

    collator = MaskedSFTCollator(tok, max_len=config.seq_len)

    class StreamDS(IterableDataset):
        def __iter__(self):
            return _make_mixed()

    # Load master weights in fp32; `fp16=True` (below) does mixed-precision via
    # autocast + GradScaler. Loading the model itself in fp16 causes
    # "Attempting to unscale FP16 gradients" with the Trainer's GradScaler.
    model = AutoModelForCausalLM.from_pretrained(config.base_model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.enable_input_require_grads()

    if config.method == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    args = TrainingArguments(
        output_dir=out_dir, max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr, fp16=True,
        optim="paged_adamw_8bit" if config.method == "full" else "adamw_torch",
        logging_steps=20, save_strategy="no", report_to=[], seed=config.seed,
    )
    Trainer(model=model, args=args, train_dataset=StreamDS(),
            data_collator=collator).train()
    # For LoRA, save_pretrained would write only the adapter (no model_type),
    # which downstream eval (run_general_eval / from_pretrained on the ckpt dir)
    # cannot load as a full model. Merge the adapter into the base first so the
    # checkpoint is an ordinary, evaluable causal-LM.
    if config.method == "lora":
        model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
