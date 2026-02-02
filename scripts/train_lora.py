# scripts/train_lora.py
import argparse, json, os
import math
import inspect
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

class FIMLineCollator:
    """
    Токенизируем prompt и completion отдельно.
    labels = -100 на prompt-части, loss только на completion.
    """
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [f["prompt"] for f in features]
        completions = [f["completion"] for f in features]

        p = self.tok(prompts, add_special_tokens=False)
        c = self.tok(completions, add_special_tokens=False)

        input_ids, attn, labels = [], [], []
        eos = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []

        for pi, ci in zip(p["input_ids"], c["input_ids"]):
            ids = pi + ci + eos
            # обрезаем справа (важнее сохранить конец префикса и completion)
            if len(ids) > self.max_length:
                overflow = len(ids) - self.max_length
                # режем prompt сначала
                if overflow < len(pi):
                    pi = pi[overflow:]
                else:
                    # если prompt слишком длинный, режем и completion тоже
                    cut = overflow - len(pi)
                    pi = []
                    ci = ci[cut:]
                ids = pi + ci + eos
                ids = ids[-self.max_length:]

            lb = ([-100] * len(pi)) + ci + eos
            lb = lb[-len(ids):]

            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attn.append(torch.ones(len(ids), dtype=torch.long))
            labels.append(torch.tensor(lb, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tok.pad_token_id)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="adapters/chatgpt4j-qwen25coder3b-lora")

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--use_4bit", action="store_true", help="QLoRA (needs bitsandbytes on Linux)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(trust_remote_code=True)
    if args.use_4bit:
        model_kwargs.update(dict(load_in_4bit=True, device_map="auto"))
    else:
        # bf16 обычно ок на A100/H100/4090; на старых GPU можно fp16
        model_kwargs.update(dict(torch_dtype=torch.bfloat16, device_map="auto"))

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Часто рекомендуют отключать кэш во время обучения (иногда предупреждения/ошибки).
    # На инференсе обратно включится по умолчанию.
    model.config.use_cache = False

    # Target-модули типичны для Qwen/Llama-подобных блоков.
    # Если вдруг имена отличаются — можно распечатать model и подправить список.
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": args.train, "validation": args.val})

    collator = FIMLineCollator(tok, max_length=args.max_length)

    # warmup_ratio deprecated -> используем warmup_steps
    # total_steps ~= ceil(N / (bs * grad_accum)) * epochs
    steps_per_epoch = math.ceil(len(ds["train"]) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(0.03 * total_steps))  # эквивалент warmup_ratio=0.03

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        logging_steps=50,
        bf16=(not args.use_4bit),
        fp16=False,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        remove_unused_columns=False,
    )

    # transformers: evaluation_strategy (old) vs eval_strategy (new)
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"

    targs = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Saved adapter to:", args.out)

if __name__ == "__main__":
    main()