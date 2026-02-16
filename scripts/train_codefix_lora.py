"""
Train a LoRA (or QLoRA) adapter for Java statement code-fix task.

Dataset format:
- Each JSONL row must contain `prompt` and `completion` fields.
- Loss is computed only on `completion` tokens; prompt tokens are ignored with label -100.

Example:
python scripts/train_codefix_lora.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --train data/codefix_train.jsonl \
  --val data/codefix_val.jsonl \
  --out adapters/codefix-qwen25coder7b-bf16-lora
"""

import argparse
import inspect
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

DEFAULT_SEED = 42


def _safe_is_bf16_supported() -> bool:
    return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())


def _resolve_dtype(arg: str, use_4bit: bool) -> torch.dtype:
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp16":
        return torch.float16
    if arg == "auto":
        if torch.cuda.is_available() and _safe_is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
    return torch.float32 if not use_4bit else torch.float16


def _enable_tf32() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]


def _set_seed(seed: int, deterministic: bool = False) -> None:
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_world_size()
        except Exception:
            return 1
    return max(1, torch.cuda.device_count())


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _log_memory(prefix: str = "") -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[VRAM]{prefix} allocated={allocated:.1f}MB reserved={reserved:.1f}MB")
    else:
        print("[VRAM] CUDA not available.")


def _print_startup_info(
    args: argparse.Namespace,
    ds: DatasetDict,
    dtype: torch.dtype,
    total_steps: int,
    warmup_steps: int,
) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    versions = {
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": getattr(__import__("transformers"), "__version__", "unknown"),
        "peft": getattr(__import__("peft"), "__version__", "unknown"),
        "datasets": getattr(__import__("datasets"), "__version__", "unknown"),
    }
    commit = _git_commit()
    effective_bs = args.batch_size * args.grad_accum * _world_size()
    print("==== Training setup ====")
    print(f"CUDA_VISIBLE_DEVICES: {visible or '<unset>'}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {device_name}")
    print(f"dtype: {dtype}, bf16={dtype == torch.bfloat16}, fp16={dtype == torch.float16}")
    print(f"gradient_checkpointing: {args.gradient_checkpointing}")
    print(
        "effective_batch_size: "
        f"{effective_bs} (per_device={args.batch_size} * grad_accum={args.grad_accum} * world={_world_size()})"
    )
    print(f"total_steps: {total_steps}, warmup_steps: {warmup_steps}")
    print(f"dataset sizes: train={len(ds['train'])}, val={len(ds['validation'])}")
    print("versions: " + ", ".join(f"{name}={ver}" for name, ver in versions.items()))
    if commit:
        print(f"git commit: {commit}")
    print("Hint: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.")
    print("========================")


def _build_bnb_config(compute_dtype: torch.dtype):
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        return None, False
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    return quant_config, True


def _choose_optimizer(use_4bit: bool, bnb_available: bool) -> str:
    if use_4bit and bnb_available:
        return "paged_adamw_8bit"
    if use_4bit and not bnb_available:
        print("Warning: bitsandbytes not available; falling back to adamw_torch.")
    return "adamw_torch"


def _select_target_modules(model, target_modules: Sequence[str]) -> Sequence[str]:
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [m for m in target_modules if m not in available]
    if missing:
        preview = sorted(list(available))[:40]
        raise ValueError(
            f"LoRA target modules not found: {missing}. "
            f"Available module names include (first 40): {preview}"
        )
    return target_modules


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class PromptCompletionCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length
        self._warned_empty = False

    def _truncate(
        self,
        prompt_ids: List[int],
        completion_ids: List[int],
        eos: List[int],
    ) -> Tuple[List[int], List[int]]:
        total = len(prompt_ids) + len(completion_ids) + len(eos)
        if total <= self.max_length:
            return prompt_ids, completion_ids

        overflow = total - self.max_length
        if overflow > 0 and prompt_ids:
            trim = min(overflow, len(prompt_ids))
            prompt_ids = prompt_ids[trim:]
            overflow -= trim

        if overflow > 0:
            if overflow >= len(completion_ids):
                completion_ids = []
            else:
                completion_ids = completion_ids[: len(completion_ids) - overflow]

        return prompt_ids, completion_ids

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [f["prompt"] for f in features]
        completions = [f["completion"] for f in features]

        tok_prompt = self.tok(prompts, add_special_tokens=False)
        tok_comp = self.tok(completions, add_special_tokens=False)

        input_ids: List[torch.Tensor] = []
        attn: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        eos = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []

        for pi, ci in zip(tok_prompt["input_ids"], tok_comp["input_ids"]):
            pi, ci = self._truncate(pi, ci, eos)
            if not ci:
                if not self._warned_empty:
                    print("Warning: dropped sample with empty completion after truncation.")
                    self._warned_empty = True
                continue

            ids = pi + ci + eos
            lbs = ([-100] * len(pi)) + ci + eos
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attn.append(torch.ones(len(ids), dtype=torch.long))
            labels.append(torch.tensor(lbs, dtype=torch.long))

        if not input_ids:
            raise ValueError("All samples in the batch were dropped after truncation.")

        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class FirstStepMemoryCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        self.logged = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.logged and state.global_step >= 1:
            self.logger(" after first step")
            self.logged = True
        return control


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train", default="data/codefix_train.jsonl")
    parser.add_argument("--val", default="data/codefix_val.jsonl")
    parser.add_argument("--out", default="adapters/codefix-qwen25coder7b-bf16-lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true", help="Enable QLoRA (bitsandbytes required).")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="bf16")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul on Ampere+ GPUs.")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing to reduce VRAM.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CuDNN (slower).",
    )
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override. Defaults to DEFAULT_SEED or SEED env.")
    parser.add_argument("--dry_run", action="store_true", help="Build model+data, run a few forward passes, then exit without saving.")
    parser.add_argument("--encoding", default="utf-8", help="File encoding for JSONL datasets.")
    return parser.parse_args()


def _filter_truncatable(example, tokenizer, max_length: int) -> bool:
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
    eos_len = 1 if tokenizer.eos_token_id is not None else 0
    total = len(prompt_ids) + len(completion_ids) + eos_len
    if total <= max_length:
        return len(completion_ids) > 0

    overflow = total - max_length
    trim = min(overflow, len(prompt_ids))
    overflow -= trim
    remaining_comp = len(completion_ids) - overflow
    return remaining_comp > 0


def _prepare_datasets(
    tokenizer,
    train_path: str,
    val_path: str,
    max_length: int,
    encoding: str,
):
    ds = load_dataset("json", data_files={"train": train_path, "validation": val_path}, encoding=encoding)
    stats = {
        "loaded": sum(len(ds[k]) for k in ds.keys()),
        "dropped_missing": 0,
    }

    def _validate(example):
        required = isinstance(example.get("prompt"), str) and isinstance(example.get("completion"), str)
        if not required:
            stats["dropped_missing"] += 1
            return False
        return True

    ds = ds.filter(_validate, load_from_cache_file=False)
    filtered = ds.filter(
        lambda ex: _filter_truncatable(ex, tokenizer, max_length),
        load_from_cache_file=False,
    )
    stats["kept"] = sum(len(filtered[k]) for k in filtered.keys())
    return filtered, stats


def _print_dataset_stats(stats: Dict) -> None:
    print("==== Dataset summary ====")
    print("dataset_format: prompt_completion")
    print(f"total_loaded: {stats.get('loaded', 0)}")
    print(f"dropped_missing_or_type: {stats.get('dropped_missing', 0)}")
    print(f"kept_after_filters: {stats.get('kept', 0)}")
    print("=========================")


def main() -> None:
    args = _parse_args()

    env_seed = os.environ.get("SEED")
    seed = args.seed if args.seed is not None else int(env_seed) if env_seed else DEFAULT_SEED
    _set_seed(seed, deterministic=args.deterministic)
    if args.tf32:
        _enable_tf32()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_dtype(args.dtype, args.use_4bit)
    if not torch.cuda.is_available() and dtype in {torch.bfloat16, torch.float16}:
        print("Warning: CUDA is not available; overriding dtype to fp32.")
        dtype = torch.float32

    quant_config = None
    bnb_available = False
    model_kwargs = dict(trust_remote_code=True, device_map="auto")
    if args.use_4bit:
        quant_config, bnb_available = _build_bnb_config(
            torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
        )
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["torch_dtype"] = torch.float16 if dtype == torch.float16 else torch.bfloat16
        else:
            print("Warning: BitsAndBytesConfig unavailable; continuing without 4-bit quantization.")
            args.use_4bit = False
            model_kwargs["torch_dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    target_modules = _select_target_modules(model, args.target_modules)

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

    datasets, ds_stats = _prepare_datasets(
        tokenizer=tokenizer,
        train_path=args.train,
        val_path=args.val,
        max_length=args.max_length,
        encoding=args.encoding,
    )
    _print_dataset_stats(ds_stats)
    collator = PromptCompletionCollator(tokenizer, max_length=args.max_length)

    steps_per_epoch = math.ceil(len(datasets["train"]) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else max(0, int(total_steps * args.warmup_ratio))
    )

    _print_startup_info(args, datasets, dtype, total_steps, warmup_steps)

    training_params = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim=_choose_optimizer(args.use_4bit, bnb_available),
        remove_unused_columns=False,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    if dtype == torch.bfloat16:
        training_kwargs["bf16"] = True
        training_kwargs["fp16"] = False
    elif dtype == torch.float16:
        training_kwargs["bf16"] = False
        training_kwargs["fp16"] = True
    else:
        training_kwargs["bf16"] = False
        training_kwargs["fp16"] = False

    def _set_strategy(keys: Sequence[str], value: str) -> None:
        for key in keys:
            if key in training_params:
                training_kwargs[key] = value
                return

    _set_strategy(("evaluation_strategy", "eval_strategy"), "steps")
    _set_strategy(("save_strategy",), "steps")
    _set_strategy(("logging_strategy",), "steps")

    training_args = TrainingArguments(**training_kwargs)

    if args.dry_run:
        print("Running dry run: building one batch and running forward passes.")
        sample = [datasets["train"][i] for i in range(min(2, len(datasets["train"])))]
        batch = collator(sample)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            _ = model(**batch)
            _log_memory(" after forward")
        print("Dry run complete; exiting without saving.")
        return

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        callbacks=[FirstStepMemoryCallback(_log_memory)],
    )

    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    _log_memory()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    run_config = {
        "args": vars(args),
        "dtype": str(dtype),
        "effective_batch_size": args.batch_size * args.grad_accum * _world_size(),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "seed": seed,
    }
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved adapter and tokenizer to {args.out}")


if __name__ == "__main__":
    main()
