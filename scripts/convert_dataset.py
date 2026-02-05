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
import os
import json

from transformers import AutoTokenizer

from train_lora import (
    DEFAULT_SEED,
    FIM_TOKENS,
    _prepare_datasets,
    _print_dataset_stats,
    _has_all_fim_tokens,
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
        "--truncate_prompt_to_max_length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Truncate prompts to fit max_seq_length while preserving FIM structure.",
    )
    ap.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Token budget for prompt + completion + eos. Defaults to --max_length.",
    )
    ap.add_argument(
        "--truncate_policy",
        choices=["left_only", "drop_file_sep_prefix_then_left"],
        default="drop_file_sep_prefix_then_left",
        help="How to shorten prompts when enforcing max_seq_length.",
    )
    ap.add_argument(
        "--min_completion_tokens",
        type=int,
        default=1,
        help="Drop samples whose completion token count is below this value.",
    )
    ap.add_argument(
        "--truncate_report_path",
        default=None,
        help="Optional JSON path to write truncation stats.",
    )
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


def _fim_tokens_in_order(prompt: str) -> bool:
    try:
        prefix_idx = prompt.index(FIM_TOKENS[0])
        suffix_idx = prompt.index(FIM_TOKENS[1], prefix_idx + len(FIM_TOKENS[0]))
        prompt.index(FIM_TOKENS[2], suffix_idx + len(FIM_TOKENS[1]))
        return True
    except ValueError:
        return False


def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _drop_file_sep_prefix_segments(prompt: str, tokenizer, budget: int):
    removed_tokens = 0
    current = prompt
    marker = "<|file_sep|>"
    while marker in current:
        start = current.find(marker)
        end = current.find(marker, start + len(marker))
        if start == -1 or end == -1:
            break
        proposal = current[:start] + current[end + len(marker) :]
        if not _has_all_fim_tokens(proposal):
            break
        current_len = _token_len(current, tokenizer)
        next_len = _token_len(proposal, tokenizer)
        removed_tokens += max(0, current_len - next_len)
        current = proposal
        if next_len <= budget:
            break
    return current, removed_tokens


def _left_trim_prompt(prompt: str, tokenizer, budget: int):
    prompt_len = _token_len(prompt, tokenizer)
    if prompt_len <= budget:
        return prompt, 0, None

    try:
        prefix_start = prompt.index(FIM_TOKENS[0])
    except ValueError:
        return None, 0, "fim_tokens_lost_after_truncation"

    tokens_before_prefix = _token_len(prompt[:prefix_start], tokenizer)
    excess = prompt_len - budget
    if excess > tokens_before_prefix:
        return None, 0, "budget_too_small_for_fim"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    trimmed_ids = prompt_ids[excess:]
    candidate = tokenizer.decode(trimmed_ids, skip_special_tokens=False)
    if not _fim_tokens_in_order(candidate):
        return None, 0, "fim_tokens_lost_after_truncation"
    return candidate, excess, None


def _apply_truncation(
    datasets,
    tokenizer,
    max_seq_length: int,
    truncate_policy: str,
    min_completion_tokens: int,
    report_path: str | None = None,
):
    stats = {
        "total_seen": 0,
        "kept_unchanged": 0,
        "truncated": 0,
        "dropped_completion_too_long": 0,
        "dropped_budget_too_small_for_fim": 0,
        "dropped_fim_tokens_lost_after_truncation": 0,
        "dropped_missing_required_fields": 0,
        "dropped_min_completion_tokens": 0,
        "average_removed_prompt_tokens": 0.0,
    }
    removed_total = 0
    eos_len = 1 if tokenizer.eos_token_id is not None else 0

    for split_name in ["train", "validation"]:
        truncated_split = []
        for ex in datasets[split_name]:
            stats["total_seen"] += 1
            prompt = ex.get("prompt")
            completion = ex.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                stats["dropped_missing_required_fields"] += 1
                continue
            completion_tokens = _token_len(completion, tokenizer)
            if completion_tokens < min_completion_tokens:
                stats["dropped_min_completion_tokens"] += 1
                continue
            budget = max_seq_length - completion_tokens - eos_len
            if budget <= 0:
                stats["dropped_completion_too_long"] += 1
                continue

            prompt_tokens = _token_len(prompt, tokenizer)
            if prompt_tokens <= budget:
                if _fim_tokens_in_order(prompt):
                    stats["kept_unchanged"] += 1
                    truncated_split.append(ex)
                else:
                    stats["dropped_fim_tokens_lost_after_truncation"] += 1
                continue

            if not _fim_tokens_in_order(prompt):
                stats["dropped_fim_tokens_lost_after_truncation"] += 1
                continue

            candidate = prompt
            removed_here = 0
            if truncate_policy == "drop_file_sep_prefix_then_left":
                candidate, removed = _drop_file_sep_prefix_segments(candidate, tokenizer, budget)
                removed_here += removed
                if _token_len(candidate, tokenizer) <= budget:
                    if _fim_tokens_in_order(candidate):
                        new_ex = dict(ex)
                        new_ex["prompt"] = candidate
                        stats["truncated"] += 1
                        removed_total += removed_here
                        truncated_split.append(new_ex)
                    else:
                        stats["dropped_fim_tokens_lost_after_truncation"] += 1
                    continue

            candidate, removed, err = _left_trim_prompt(candidate, tokenizer, budget)
            if err:
                if err == "budget_too_small_for_fim":
                    stats["dropped_budget_too_small_for_fim"] += 1
                else:
                    stats["dropped_fim_tokens_lost_after_truncation"] += 1
                continue

            removed_here += removed
            if _token_len(candidate, tokenizer) > budget:
                stats["dropped_budget_too_small_for_fim"] += 1
                continue
            if not _fim_tokens_in_order(candidate):
                stats["dropped_fim_tokens_lost_after_truncation"] += 1
                continue

            new_ex = dict(ex)
            new_ex["prompt"] = candidate
            stats["truncated"] += 1
            removed_total += removed_here
            truncated_split.append(new_ex)

        datasets[split_name] = truncated_split

    if stats["truncated"]:
        stats["average_removed_prompt_tokens"] = removed_total / stats["truncated"]
    if report_path:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def _print_truncation_summary(stats: dict):
    dropped = (
        stats.get("dropped_completion_too_long", 0)
        + stats.get("dropped_budget_too_small_for_fim", 0)
        + stats.get("dropped_fim_tokens_lost_after_truncation", 0)
        + stats.get("dropped_missing_required_fields", 0)
        + stats.get("dropped_min_completion_tokens", 0)
    )
    avg_removed = stats.get("average_removed_prompt_tokens", 0.0)
    print(
        "Truncation summary: "
        f"seen={stats.get('total_seen', 0)} "
        f"kept={stats.get('kept_unchanged', 0)} "
        f"truncated={stats.get('truncated', 0)} "
        f"dropped={dropped} "
        f"(completion_too_long={stats.get('dropped_completion_too_long', 0)}, "
        f"budget_too_small_for_fim={stats.get('dropped_budget_too_small_for_fim', 0)}, "
        f"fim_tokens_lost={stats.get('dropped_fim_tokens_lost_after_truncation', 0)}, "
        f"min_completion_tokens={stats.get('dropped_min_completion_tokens', 0)}, "
        f"missing_required_fields={stats.get('dropped_missing_required_fields', 0)}) "
        f"avg_removed_prompt_tokens={avg_removed:.2f}"
    )


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

    max_seq_length = args.max_seq_length or args.max_length
    truncation_stats = None
    if args.truncate_prompt_to_max_length:
        truncation_stats = _apply_truncation(
            datasets=datasets,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            truncate_policy=args.truncate_policy,
            min_completion_tokens=args.min_completion_tokens,
            report_path=args.truncate_report_path,
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

    if truncation_stats:
        _print_truncation_summary(truncation_stats)
    _print_examples(datasets["train"], tokenizer, args.example_count)


if __name__ == "__main__":
    main()
