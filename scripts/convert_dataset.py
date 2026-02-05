"""
Smoke-test and convert FIM datasets to the prompt/completion schema used by train_lora.py.

Usage:
python scripts/convert_dataset.py \
  --train data/train_3fields.jsonl \
  --val data/val_3fields.jsonl \
  --out_train data/converted_train.jsonl \
  --out_val data/converted_val.jsonl \
  --dataset_format fim_expected_completed
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from train_lora import (
    DEFAULT_SEED,
    _prepare_datasets,
    _print_dataset_stats,
    _set_seed,
)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out_train", default="data/converted_train.jsonl")
    ap.add_argument("--out_val", default="data/converted_val.jsonl")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument(
        "--dataset_format",
        choices=["fim_native", "fim_expected_completed"],
        default="fim_native",
    )
    ap.add_argument(
        "--require_fim_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require all FIM tokens (<|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>) in prompt.",
    )
    ap.add_argument(
        "--keep_completed_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep expectedCode/completedCode fields for analysis when using 'fim_expected_completed'.",
    )
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--example_count", type=int, default=2, help="How many samples to preview.")
    return ap.parse_args()


def _write_jsonl(dataset_split, path: str, encoding: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        for row in dataset_split:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _print_examples(split, tokenizer, example_count: int):
    print("==== Example snippets ====")
    for i in range(min(example_count, len(split))):
        ex = split[i]
        prompt_tokens = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        completion_tokens = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
        print(f"[{i}] prompt_chars={len(ex['prompt'])} completion_chars={len(ex['completion'])} "
              f"prompt_tokens={len(prompt_tokens)} completion_tokens={len(completion_tokens)}")
    print("=========================")


def main():
    args = _parse_args()
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    _set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets, stats = _prepare_datasets(
        tokenizer=tokenizer,
        train_path=args.train,
        val_path=args.val,
        max_length=args.max_length,
        dataset_format=args.dataset_format,
        require_fim_tokens=args.require_fim_tokens,
        keep_completed_code=args.keep_completed_code,
        encoding=args.encoding,
    )

    _print_dataset_stats(stats, args.dataset_format)
    print(f"train size: {len(datasets['train'])}")
    print(f"val size: {len(datasets['validation'])}")

    if args.out_train:
        _write_jsonl(datasets["train"], args.out_train, args.encoding)
        print(f"Wrote converted train split to {args.out_train}")
    if args.out_val:
        _write_jsonl(datasets["validation"], args.out_val, args.encoding)
        print(f"Wrote converted val split to {args.out_val}")

    _print_examples(datasets["train"], tokenizer, args.example_count)


if __name__ == "__main__":
    main()
