"""
Smoke-test and convert FIM datasets to the prompt/completion schema used by train_lora.py.
Now with progress logging, timing diagnostics, truncation safeguards, and a "pure FIM"
mode that strips <|file_sep|> blocks from augmented prompts.

Examples:
- Basic conversion: python scripts/convert_dataset.py --dataset_format fim_expected_completed
- Strip to pure FIM: python scripts/convert_dataset.py --strip_to_pure_fim --report_path data/convert_report.json
- Strip + truncate: python scripts/convert_dataset.py --strip_to_pure_fim --truncate_prompt_to_max_length --max_seq_length 2048
"""

import argparse
import json
import os
import time
from collections import defaultdict
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

from transformers import AutoTokenizer

from train_lora import (
    DEFAULT_SEED,
    FIM_TOKENS,
    _has_all_fim_tokens,
    _prepare_datasets,
    _print_dataset_stats,
    _set_seed,
)

LOG_LEVELS = {"quiet": 0, "normal": 1, "verbose": 2}
FILE_SEP = "<|file_sep|>"
UNKNOWN_MODEL_MAX_LENGTH = 1_000_000

DROP_REASONS = {
    "missing_required_fields": "missing_required_fields",
    "min_completion_tokens": "min_completion_tokens",
    "completion_too_long": "completion_too_long",
    "fim_tokens_lost_after_truncation": "fim_tokens_lost_after_truncation",
    "fim_tokens_lost_after_strip": "fim_tokens_lost_after_strip",
    "budget_too_small_for_fim": "budget_too_small_for_fim",
    "prompt_exceeds_model_max_length": "prompt_exceeds_model_max_length",
    "hit_file_sep_max_iters": "hit_file_sep_max_iters",
    "hit_max_tokenize_calls_per_example": "hit_max_tokenize_calls_per_example",
    "skipped_file_sep_pass_due_to_char_len": "skipped_file_sep_pass_due_to_char_len",
    "rag_variant_requires_truncation": "rag_variant_requires_truncation",
    "both_mode_conflict_skip": "both_mode_conflict_skip",
}


class TokenizeLimitError(Exception):
    """Raised when per-example tokenize call budget is exceeded."""


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
        help="Optional JSON path to write truncation stats (legacy; now superseded by --report_path).",
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
    ap.add_argument("--log_every_n", type=int, default=200, help="Emit progress every N examples.")
    ap.add_argument(
        "--log_level",
        choices=["quiet", "normal", "verbose"],
        default="normal",
        help="Logging verbosity.",
    )
    ap.add_argument(
        "--time_breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect timing breakdown (tokenization, stripping, truncation).",
    )
    ap.add_argument(
        "--file_sep_max_iters",
        type=int,
        default=128,
        help="Maximum <|file_sep|> removal iterations per example.",
    )
    ap.add_argument(
        "--max_prompt_char_len_for_file_sep_pass",
        type=int,
        default=300000,
        help="Skip iterative file-sep trimming when prompts exceed this character length.",
    )
    ap.add_argument(
        "--max_tokenize_calls_per_example",
        type=int,
        default=50,
        help="Maximum tokenizer calls per example before triggering a safeguard.",
    )
    ap.add_argument(
        "--drop_on_safeguard_hit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop examples that hit safeguard limits. If False, keep them unchanged when possible.",
    )
    ap.add_argument(
        "--strip_to_pure_fim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove all <|file_sep|> blocks to keep only the core FIM prompt.",
    )
    ap.add_argument(
        "--strip_file_sep_mode",
        choices=["remove_all_pairs", "remove_prefix_only"],
        default="remove_all_pairs",
        help="Mode for stripping <|file_sep|> blocks when --strip_to_pure_fim is set.",
    )
    ap.add_argument(
        "--strip_report",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include strip stats in the JSON report (defaults to True when strip is enabled).",
    )
    ap.add_argument(
        "--token_len_cache_size",
        type=int,
        default=4096,
        help="LRU cache size for token length computation.",
    )
    ap.add_argument(
        "--file_sep_retokenize_every",
        type=int,
        default=8,
        help="Re-tokenize every K file-sep removals; otherwise rely on char-length heuristics.",
    )
    ap.add_argument(
        "--enforce_model_max_length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop or truncate examples that exceed tokenizer.model_max_length.",
    )
    ap.add_argument(
        "--emit_both_rag_and_norag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit both No-RAG and RAG variants for each example when possible.",
    )
    ap.add_argument(
        "--both_mode_suffixes",
        default="norag,rag",
        help="Comma-separated variant suffixes to tag No-RAG and RAG outputs.",
    )
    ap.add_argument(
        "--both_mode_on_conflict",
        choices=["skip", "keep_one"],
        default="keep_one",
        help="When only one variant succeeds, either keep it or skip both.",
    )
    ap.add_argument(
        "--both_mode_require_truncation_for_rag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require truncation flags to be set for emitting the RAG variant.",
    )
    ap.add_argument(
        "--report_path",
        default=None,
        help="Optional JSON path to write conversion report (supersedes --truncate_report_path).",
    )
    return ap.parse_args()


def _write_jsonl(dataset_split, path: str, encoding: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        for row in dataset_split:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _print_examples(split, tokenizer, example_count: int, log_level: str):
    if LOG_LEVELS[log_level] == LOG_LEVELS["quiet"]:
        return
    print("==== Example snippets ====")
    for i in range(min(example_count, len(split))):
        ex = split[i]
        prompt_tokens = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        completion_tokens = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
        print(
            f"[{i}] prompt_chars={len(ex['prompt'])} completion_chars={len(ex['completion'])} "
            f"prompt_tokens={len(prompt_tokens)} completion_tokens={len(completion_tokens)}"
        )
    print("=========================")


def _fim_tokens_in_order(prompt: str) -> bool:
    try:
        prefix_idx = prompt.index(FIM_TOKENS[0])
        suffix_idx = prompt.index(FIM_TOKENS[1], prefix_idx + len(FIM_TOKENS[0]))
        prompt.index(FIM_TOKENS[2], suffix_idx + len(FIM_TOKENS[1]))
        return True
    except ValueError:
        return False


def _build_token_len_fn(tokenizer, cache_size: int) -> Callable[[str], int]:
    @lru_cache(maxsize=cache_size)
    def _cached(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return _cached


class TokenLenHelper:
    def __init__(
        self,
        tokenizer,
        cache_size: int,
        max_calls_per_example: int,
        timing: Dict[str, float] | None,
    ):
        self.tokenizer = tokenizer
        self._cached = _build_token_len_fn(tokenizer, cache_size)
        self.max_calls_per_example = max_calls_per_example
        self.example_calls = 0
        self.total_calls = 0
        self._timing = timing
        self._last_text: str | None = None
        self._last_tokens = None

    def reset_example(self):
        self.example_calls = 0
        self._last_text = None
        self._last_tokens = None

    def _tick_and_time(self):
        self.example_calls += 1
        self.total_calls += 1
        if self.example_calls > self.max_calls_per_example:
            raise TokenizeLimitError()
        start = time.perf_counter() if self._timing is not None else None
        return start

    def _maybe_record_time(self, start: float | None):
        if start is not None:
            self._timing["tokenization"] = self._timing.get("tokenization", 0.0) + (
                time.perf_counter() - start
            )

    def token_len(self, text: str) -> int:
        start = self._tick_and_time()
        result = self._cached(text)
        self._maybe_record_time(start)
        return result

    def tokens(self, text: str):
        start = self._tick_and_time()
        if text == self._last_text and self._last_tokens is not None:
            self._maybe_record_time(start)
            return self._last_tokens
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        self._maybe_record_time(start)
        self._last_text = text
        self._last_tokens = ids
        return ids


def _init_split_stats() -> Dict:
    return {
        "total_seen": 0,
        "originals_seen": 0,
        "originals_emitted_2": 0,
        "originals_emitted_1": 0,
        "originals_emitted_0": 0,
        "variants_emitted_total": 0,
        "norag_emitted": 0,
        "rag_emitted": 0,
        "variant_emitted_counts": defaultdict(int),
        "drop_reasons_by_variant": defaultdict(lambda: defaultdict(int)),
        "kept": 0,
        "truncated": 0,
        "stripped": 0,
        "dropped": 0,
        "drop_reasons": defaultdict(int),
        "safeguards": {
            "hit_file_sep_max_iters": 0,
            "hit_max_tokenize_calls_per_example": 0,
            "skipped_file_sep_pass_due_to_char_len": 0,
            "unmatched_file_sep_marker": 0,
        },
        "timing": defaultdict(float),
        "tokenize_calls": 0,
        "prompt_tokens_sum_before": 0,
        "prompt_tokens_sum_after": 0,
        "prompt_tokens_max_before": 0,
        "prompt_tokens_max_after": 0,
        "prompt_tokens_over_model_max": 0,
        "completion_tokens_sum": 0,
        "completion_tokens_max": 0,
        "removed_prompt_tokens_total": 0,
        "file_sep_segments_removed": 0,
        "file_sep_tokens_removed_estimate": 0,
        "examples_with_prompt_tokens_over_budget": 0,
    }


def _merge_global_stats(per_split: Dict[str, Dict]) -> Dict:
    merged = _init_split_stats()
    for split_stats in per_split.values():
        merged["total_seen"] += split_stats["total_seen"]
        merged["originals_seen"] += split_stats["originals_seen"]
        merged["originals_emitted_2"] += split_stats["originals_emitted_2"]
        merged["originals_emitted_1"] += split_stats["originals_emitted_1"]
        merged["originals_emitted_0"] += split_stats["originals_emitted_0"]
        merged["variants_emitted_total"] += split_stats["variants_emitted_total"]
        merged["norag_emitted"] += split_stats["norag_emitted"]
        merged["rag_emitted"] += split_stats["rag_emitted"]
        for variant, count in split_stats["variant_emitted_counts"].items():
            merged["variant_emitted_counts"][variant] += count
        merged["kept"] += split_stats["kept"]
        merged["truncated"] += split_stats["truncated"]
        merged["stripped"] += split_stats["stripped"]
        merged["dropped"] += split_stats["dropped"]
        merged["tokenize_calls"] += split_stats["tokenize_calls"]
        merged["prompt_tokens_sum_before"] += split_stats["prompt_tokens_sum_before"]
        merged["prompt_tokens_sum_after"] += split_stats["prompt_tokens_sum_after"]
        merged["prompt_tokens_max_before"] = max(
            merged["prompt_tokens_max_before"], split_stats["prompt_tokens_max_before"]
        )
        merged["prompt_tokens_max_after"] = max(
            merged["prompt_tokens_max_after"], split_stats["prompt_tokens_max_after"]
        )
        merged["prompt_tokens_over_model_max"] += split_stats["prompt_tokens_over_model_max"]
        merged["completion_tokens_sum"] += split_stats["completion_tokens_sum"]
        merged["completion_tokens_max"] = max(
            merged["completion_tokens_max"], split_stats["completion_tokens_max"]
        )
        merged["removed_prompt_tokens_total"] += split_stats["removed_prompt_tokens_total"]
        merged["file_sep_segments_removed"] += split_stats["file_sep_segments_removed"]
        merged["file_sep_tokens_removed_estimate"] += split_stats["file_sep_tokens_removed_estimate"]
        merged["examples_with_prompt_tokens_over_budget"] += split_stats[
            "examples_with_prompt_tokens_over_budget"
        ]
        for k, v in split_stats["drop_reasons"].items():
            merged["drop_reasons"][k] += v
        for variant, reasons in split_stats["drop_reasons_by_variant"].items():
            for reason, count in reasons.items():
                merged["drop_reasons_by_variant"][variant][reason] += count
        for k, v in split_stats["safeguards"].items():
            merged["safeguards"][k] += v
        for k, v in split_stats["timing"].items():
            merged["timing"][k] += v
    return merged


def _log(msg: str, level: str, log_level: str):
    if LOG_LEVELS[log_level] >= LOG_LEVELS[level]:
        print(msg)


def _record_drop_reason(stats: Dict, reason: str, variant: str | None = None):
    stats["drop_reasons"][reason] += 1
    if variant is not None:
        stats["drop_reasons_by_variant"][variant][reason] += 1


def _strip_file_sep_blocks(
    prompt: str,
    token_helper: TokenLenHelper,
    args,
    mode: str,
    stats: Dict,
) -> Tuple[str, Dict]:
    """
    Remove <|file_sep|> blocks either everywhere or before the first FIM prefix.
    Returns (new_prompt, meta) where meta carries counters and safeguard flags.
    """
    iterations = 0
    segments_removed = 0
    unmatched = 0
    current = prompt
    approx_len = token_helper.token_len(current)
    initial_len = approx_len

    while FILE_SEP in current:
        if iterations >= args.file_sep_max_iters:
            stats["safeguards"]["hit_file_sep_max_iters"] += 1
            final_len_guard = token_helper.token_len(current)
            return current, {
                "hit_max_iters": True,
                "segments_removed": segments_removed,
                "removed_tokens": max(0, initial_len - final_len_guard),
                "unmatched": unmatched,
                "final_len": final_len_guard,
            }

        try:
            prefix_idx = current.index(FIM_TOKENS[0])
        except ValueError:
            prefix_idx = -1

        start = current.find(FILE_SEP)
        end = current.find(FILE_SEP, start + len(FILE_SEP))
        if start == -1:
            break
        if end == -1:
            unmatched += 1
            break
        if mode == "remove_prefix_only" and prefix_idx != -1 and start >= prefix_idx:
            break
        proposal = current[:start] + current[end + len(FILE_SEP) :]
        if not _has_all_fim_tokens(proposal):
            break

        iterations += 1
        segments_removed += 1
        ratio = len(proposal) / len(current) if current else 0
        approx_len = max(0, int(approx_len * ratio))
        if iterations % args.file_sep_retokenize_every == 0:
            approx_len = token_helper.token_len(proposal)

        current = proposal
        if approx_len <= 0:
            break

    final_len = token_helper.token_len(current)
    removed_tokens = max(0, initial_len - final_len)
    stats["safeguards"]["unmatched_file_sep_marker"] += unmatched
    stats["file_sep_segments_removed"] += segments_removed
    stats["file_sep_tokens_removed_estimate"] += removed_tokens
    return current, {
        "hit_max_iters": iterations >= args.file_sep_max_iters,
        "segments_removed": segments_removed,
        "removed_tokens": removed_tokens,
        "unmatched": unmatched,
        "final_len": final_len,
    }


def _left_trim_prompt(prompt: str, token_helper: TokenLenHelper, budget: int, stats: Dict):
    start = time.perf_counter()
    prompt_len = token_helper.token_len(prompt)
    if prompt_len <= budget:
        stats["timing"]["left_trim_time"] += time.perf_counter() - start
        return prompt, 0, None

    try:
        prefix_start = prompt.index(FIM_TOKENS[0])
    except ValueError:
        stats["timing"]["left_trim_time"] += time.perf_counter() - start
        return None, 0, DROP_REASONS["fim_tokens_lost_after_truncation"]

    tokens_before_prefix = token_helper.token_len(prompt[:prefix_start])
    excess = prompt_len - budget
    if excess > tokens_before_prefix:
        stats["timing"]["left_trim_time"] += time.perf_counter() - start
        return None, 0, DROP_REASONS["budget_too_small_for_fim"]

    prompt_ids = token_helper.tokens(prompt)
    trimmed_ids = prompt_ids[excess:]
    candidate = token_helper.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
    if not _fim_tokens_in_order(candidate):
        stats["timing"]["left_trim_time"] += time.perf_counter() - start
        return None, 0, DROP_REASONS["fim_tokens_lost_after_truncation"]
    stats["timing"]["left_trim_time"] += time.perf_counter() - start
    return candidate, excess, None


def _truncate_prompt(
    prompt: str,
    completion_tokens: int,
    eos_len: int,
    token_helper: TokenLenHelper,
    args,
    stats: Dict,
) -> Tuple[str | None, int | None, str | None]:
    """Apply truncation policy. Returns (new_prompt, removed_tokens, drop_reason)."""
    trunc_start = time.perf_counter()
    budget = args.max_seq_length - completion_tokens - eos_len
    if budget <= 0:
        return None, None, DROP_REASONS["completion_too_long"]

    prompt_tokens = token_helper.token_len(prompt)
    if prompt_tokens <= budget:
        stats["timing"]["truncation_total"] += time.perf_counter() - trunc_start
        return prompt, 0, None

    if not _fim_tokens_in_order(prompt):
        stats["timing"]["truncation_total"] += time.perf_counter() - trunc_start
        return None, None, DROP_REASONS["fim_tokens_lost_after_truncation"]

    candidate = prompt
    removed_tokens = 0
    if (
        args.truncate_policy == "drop_file_sep_prefix_then_left"
        and len(candidate) <= args.max_prompt_char_len_for_file_sep_pass
    ):
        strip_start = time.perf_counter()
        candidate, meta = _strip_file_sep_blocks(
            candidate, token_helper, args, mode="remove_prefix_only", stats=stats
        )
        stats["timing"]["file_sep_stripping_time"] += time.perf_counter() - strip_start
        if meta.get("hit_max_iters"):
            stats["timing"]["truncation_total"] += time.perf_counter() - trunc_start
            return None, None, DROP_REASONS["hit_file_sep_max_iters"]
        removed_tokens = meta["removed_tokens"]
    elif args.truncate_policy == "drop_file_sep_prefix_then_left":
        stats["safeguards"]["skipped_file_sep_pass_due_to_char_len"] += 1

    candidate, removed_left, err = _left_trim_prompt(candidate, token_helper, budget, stats)

    if err:
        stats["timing"]["truncation_total"] += time.perf_counter() - trunc_start
        return None, None, err

    removed_tokens += removed_left
    if token_helper.token_len(candidate) > budget:
        stats["timing"]["truncation_total"] += time.perf_counter() - trunc_start
        return None, None, DROP_REASONS["budget_too_small_for_fim"]
    if not _fim_tokens_in_order(candidate):
        stats["timing"]["truncation_total"] += time.perf_counter() - trunc_start
        return None, None, DROP_REASONS["fim_tokens_lost_after_truncation"]

    stats["timing"]["truncation_total"] += time.perf_counter() - trunc_start
    return candidate, removed_tokens, None


def _process_split(
    split_name: str,
    examples,
    tokenizer,
    token_helper: TokenLenHelper,
    args,
    model_max_length: int | None,
) -> Tuple[list, Dict]:
    stats = _init_split_stats()
    processed = []
    start_time = time.perf_counter()
    eos_len = 1 if tokenizer.eos_token_id is not None else 0

    if args.emit_both_rag_and_norag:
        norag_label, rag_label = args.both_mode_suffixes_list
        variant_configs: List[Dict] = [
            {
                "name": norag_label,
                "role": "norag",
                "apply_strip": True,
                "strip_mode": "remove_all_pairs",
                "require_truncation": False,
            },
            {
                "name": rag_label,
                "role": "rag",
                "apply_strip": False,
                "strip_mode": args.strip_file_sep_mode,
                "require_truncation": args.both_mode_require_truncation_for_rag,
            },
        ]
    else:
        variant_configs = [
            {
                "name": "single",
                "role": "single",
                "apply_strip": args.strip_to_pure_fim,
                "strip_mode": args.strip_file_sep_mode,
                "require_truncation": False,
            }
        ]

    def _finalize_variant_result(res: Dict, emit: bool):
        stats["total_seen"] += 1
        stats["tokenize_calls"] += res.get("token_calls", 0)
        completion_tokens = res.get("completion_tokens")
        if completion_tokens is not None:
            stats["completion_tokens_sum"] += completion_tokens
            stats["completion_tokens_max"] = max(stats["completion_tokens_max"], completion_tokens)
        if res.get("prompt_tokens_before") is not None:
            stats["prompt_tokens_sum_before"] += res["prompt_tokens_before"]
            stats["prompt_tokens_max_before"] = max(
                stats["prompt_tokens_max_before"], res["prompt_tokens_before"]
            )
        if res.get("model_max_hit"):
            stats["prompt_tokens_over_model_max"] += 1
        if res.get("prompt_tokens_over_budget"):
            stats["examples_with_prompt_tokens_over_budget"] += 1
        if res.get("stripped"):
            stats["stripped"] += 1

        variant_name = res["name"]
        if not emit or res.get("drop_reason"):
            stats["dropped"] += 1
            _record_drop_reason(
                stats,
                res.get("drop_reason", DROP_REASONS["both_mode_conflict_skip"]),
                variant_name,
            )
            return

        if res.get("prompt_tokens_after") is not None:
            stats["prompt_tokens_sum_after"] += res["prompt_tokens_after"]
            stats["prompt_tokens_max_after"] = max(
                stats["prompt_tokens_max_after"], res["prompt_tokens_after"]
            )
        if res.get("truncated"):
            stats["truncated"] += 1
            stats["removed_prompt_tokens_total"] += res.get("removed_tokens") or 0
        else:
            stats["kept"] += 1

        stats["variants_emitted_total"] += 1
        stats["variant_emitted_counts"][variant_name] += 1
        if res.get("role") == "norag":
            stats["norag_emitted"] += 1
        if res.get("role") == "rag":
            stats["rag_emitted"] += 1
        processed.append(res["example"])

    def _attempt_variant(
        base_prompt: str,
        completion_tokens: int,
        variant_cfg: Dict,
        source_id: str,
    ) -> Dict:
        variant_calls_before = token_helper.example_calls
        prompt = base_prompt
        prompt_tokens_before = None
        prompt_tokens_after = None
        model_max_hit = False
        prompt_tokens_over_budget = False
        removed_tokens = 0
        truncated = False
        drop_reason = None
        stripped = False

        if variant_cfg["role"] == "rag" and variant_cfg["require_truncation"]:
            if not args.truncate_prompt_to_max_length:
                return {
                    "name": variant_cfg["name"],
                    "role": variant_cfg["role"],
                    "drop_reason": DROP_REASONS["rag_variant_requires_truncation"],
                    "token_calls": token_helper.example_calls - variant_calls_before,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens_before": None,
                    "prompt_tokens_after": None,
                    "model_max_hit": False,
                    "prompt_tokens_over_budget": False,
                    "truncated": False,
                    "removed_tokens": 0,
                    "stripped": False,
                }

        try:
            prompt_tokens_before = token_helper.token_len(prompt)
        except TokenizeLimitError:
            stats["safeguards"]["hit_max_tokenize_calls_per_example"] += 1
            if args.drop_on_safeguard_hit:
                drop_reason = DROP_REASONS["hit_max_tokenize_calls_per_example"]
            else:
                prompt_tokens_before = None

        if prompt_tokens_before is not None and model_max_length is not None:
            if prompt_tokens_before > model_max_length:
                stats["prompt_tokens_over_model_max"] += 1
                model_max_hit = True

        if variant_cfg["apply_strip"]:
            stripped = True
            strip_start = time.perf_counter()
            try:
                prompt, meta = _strip_file_sep_blocks(
                    prompt, token_helper, args, mode=variant_cfg["strip_mode"], stats=stats
                )
            except TokenizeLimitError:
                stats["safeguards"]["hit_max_tokenize_calls_per_example"] += 1
                if args.drop_on_safeguard_hit:
                    drop_reason = DROP_REASONS["hit_max_tokenize_calls_per_example"]
                else:
                    prompt = base_prompt
                    meta = {"hit_max_iters": False, "segments_removed": 0, "removed_tokens": 0}
            stats["timing"]["file_sep_stripping_time"] += time.perf_counter() - strip_start
            if meta.get("hit_max_iters"):
                if args.drop_on_safeguard_hit:
                    drop_reason = DROP_REASONS["hit_file_sep_max_iters"]
                else:
                    prompt = base_prompt
            elif isinstance(meta.get("final_len"), int):
                prompt_tokens_before = meta["final_len"]
            if not drop_reason and not _fim_tokens_in_order(prompt):
                drop_reason = DROP_REASONS["fim_tokens_lost_after_strip"]

        if drop_reason is None and args.truncate_prompt_to_max_length:
            budget = args.max_seq_length - completion_tokens - eos_len
            if prompt_tokens_before is not None and prompt_tokens_before > budget:
                prompt_tokens_over_budget = True
            try:
                truncated_prompt, removed_tokens, err = _truncate_prompt(
                    prompt,
                    completion_tokens,
                    eos_len,
                    token_helper,
                    args,
                    stats,
                )
            except TokenizeLimitError:
                stats["safeguards"]["hit_max_tokenize_calls_per_example"] += 1
                err = DROP_REASONS["hit_max_tokenize_calls_per_example"]

            if err:
                safeguard_reasons = {
                    DROP_REASONS["hit_max_tokenize_calls_per_example"],
                    DROP_REASONS["hit_file_sep_max_iters"],
                }
                if err in safeguard_reasons and not args.drop_on_safeguard_hit:
                    truncated_prompt = base_prompt
                    err = None
                    removed_tokens = 0
                else:
                    drop_reason = err
            else:
                truncated = bool(removed_tokens)
                prompt = truncated_prompt
                if not _fim_tokens_in_order(prompt):
                    drop_reason = DROP_REASONS["fim_tokens_lost_after_truncation"]

        prompt_tokens_after = prompt_tokens_before
        if drop_reason is None:
            try:
                prompt_tokens_after = token_helper.token_len(prompt)
            except TokenizeLimitError:
                stats["safeguards"]["hit_max_tokenize_calls_per_example"] += 1
                if args.drop_on_safeguard_hit:
                    drop_reason = DROP_REASONS["hit_max_tokenize_calls_per_example"]
                else:
                    prompt_tokens_after = prompt_tokens_before

        if (
            drop_reason is None
            and args.enforce_model_max_length
            and model_max_length is not None
            and prompt_tokens_after is not None
            and prompt_tokens_after > model_max_length
        ):
            drop_reason = DROP_REASONS["prompt_exceeds_model_max_length"]

        if drop_reason is None and not _fim_tokens_in_order(prompt):
            drop_reason = DROP_REASONS["fim_tokens_lost_after_truncation"]

        variant_calls = token_helper.example_calls - variant_calls_before
        if drop_reason is None and prompt_tokens_after is None:
            prompt_tokens_after = prompt_tokens_before or 0

        result_example = None
        if drop_reason is None:
            new_ex = dict(ex)
            new_ex["prompt"] = prompt
            new_ex["variant"] = variant_cfg["name"]
            new_ex["source_id"] = source_id
            new_ex["variant_id"] = f"{source_id}::{variant_cfg['name']}"
            result_example = new_ex

        return {
            "name": variant_cfg["name"],
            "role": variant_cfg["role"],
            "drop_reason": drop_reason,
            "prompt_tokens_before": prompt_tokens_before,
            "prompt_tokens_after": prompt_tokens_after,
            "removed_tokens": removed_tokens,
            "truncated": truncated,
            "model_max_hit": model_max_hit,
            "prompt_tokens_over_budget": prompt_tokens_over_budget,
            "token_calls": variant_calls,
            "completion_tokens": completion_tokens,
            "example": result_example,
            "stripped": stripped,
        }

    for idx, ex in enumerate(examples, start=1):
        token_helper.reset_example()
        stats["originals_seen"] += 1
        prompt = ex.get("prompt")
        completion = ex.get("completion")
        source_id = ex.get("id") or ex.get("source_id") or f"{split_name}:{idx}"

        if not isinstance(prompt, str) or not isinstance(completion, str):
            for variant_cfg in variant_configs:
                _finalize_variant_result(
                    {
                        "name": variant_cfg["name"],
                        "role": variant_cfg["role"],
                        "drop_reason": DROP_REASONS["missing_required_fields"],
                        "prompt_tokens_before": None,
                        "prompt_tokens_after": None,
                        "removed_tokens": 0,
                        "truncated": False,
                        "model_max_hit": False,
                        "prompt_tokens_over_budget": False,
                        "token_calls": token_helper.example_calls,
                        "completion_tokens": None,
                        "example": None,
                        "stripped": variant_cfg["apply_strip"],
                    },
                    emit=False,
                )
            stats["originals_emitted_0"] += 1
            continue

        try:
            completion_tokens = token_helper.token_len(completion)
            completion_calls = token_helper.example_calls
        except TokenizeLimitError:
            stats["safeguards"]["hit_max_tokenize_calls_per_example"] += 1
            if args.drop_on_safeguard_hit:
                for variant_cfg in variant_configs:
                    _finalize_variant_result(
                        {
                            "name": variant_cfg["name"],
                            "role": variant_cfg["role"],
                            "drop_reason": DROP_REASONS["hit_max_tokenize_calls_per_example"],
                            "prompt_tokens_before": None,
                            "prompt_tokens_after": None,
                            "removed_tokens": 0,
                            "truncated": False,
                            "model_max_hit": False,
                            "prompt_tokens_over_budget": False,
                            "token_calls": token_helper.example_calls,
                            "completion_tokens": None,
                            "example": None,
                            "stripped": variant_cfg["apply_strip"],
                        },
                        emit=False,
                    )
                stats["originals_emitted_0"] += 1
                continue
            completion_tokens = None
            completion_calls = token_helper.example_calls

        stats["tokenize_calls"] += completion_calls

        if completion_tokens is not None and completion_tokens < args.min_completion_tokens:
            for variant_cfg in variant_configs:
                _finalize_variant_result(
                    {
                        "name": variant_cfg["name"],
                        "role": variant_cfg["role"],
                        "drop_reason": DROP_REASONS["min_completion_tokens"],
                        "prompt_tokens_before": None,
                        "prompt_tokens_after": None,
                        "removed_tokens": 0,
                        "truncated": False,
                        "model_max_hit": False,
                        "prompt_tokens_over_budget": False,
                        "token_calls": 0,
                        "completion_tokens": completion_tokens,
                        "example": None,
                        "stripped": variant_cfg["apply_strip"],
                    },
                    emit=False,
                )
            stats["originals_emitted_0"] += 1
            continue

        variant_results = []
        for variant_cfg in variant_configs:
            result = _attempt_variant(prompt, completion_tokens or 0, variant_cfg, source_id)
            variant_results.append(result)

        emitted_results = []
        failed_results = []
        for res in variant_results:
            if res.get("drop_reason"):
                failed_results.append(res)
            else:
                emitted_results.append(res)

        if args.emit_both_rag_and_norag and args.both_mode_on_conflict == "skip":
            if emitted_results and failed_results:
                for res in emitted_results:
                    res["drop_reason"] = DROP_REASONS["both_mode_conflict_skip"]
                failed_results.extend(emitted_results)
                emitted_results = []

        for res in emitted_results:
            _finalize_variant_result(res, emit=True)
        for res in failed_results:
            _finalize_variant_result(res, emit=False)

        emitted_count = len(emitted_results)
        if len(variant_configs) == 1:
            if emitted_count == 1:
                stats["originals_emitted_1"] += 1
            else:
                stats["originals_emitted_0"] += 1
        else:
            if emitted_count == len(variant_configs):
                stats["originals_emitted_2"] += 1
            elif emitted_count == 1:
                stats["originals_emitted_1"] += 1
            else:
                stats["originals_emitted_0"] += 1

        if idx % args.log_every_n == 0 and LOG_LEVELS[args.log_level] >= LOG_LEVELS["normal"]:
            elapsed = time.perf_counter() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            _log(
                f"[convert] split={split_name} idx={idx}/{len(examples)} "
                f"variants_emitted_total={stats['variants_emitted_total']} dropped={stats['dropped']} "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} ex/s",
                "normal",
                args.log_level,
            )

    elapsed = time.perf_counter() - start_time
    rate = stats["total_seen"] / elapsed if elapsed > 0 else 0.0
    _log(
        f"[convert] split={split_name} done "
        f"variants_emitted={stats['variants_emitted_total']} dropped={stats['dropped']} "
        f"elapsed={elapsed:.1f}s rate={rate:.1f} ex/s",
        "normal",
        args.log_level,
    )
    if args.emit_both_rag_and_norag:
        _log(
            f"[convert] split={split_name} originals={stats['originals_seen']} "
            f"emitted_total={stats['variants_emitted_total']} "
            f"(norag={stats['norag_emitted']} rag={stats['rag_emitted']}) "
            f"originals:2={stats['originals_emitted_2']} "
            f"1={stats['originals_emitted_1']} 0={stats['originals_emitted_0']}",
            "normal",
            args.log_level,
        )
    return processed, stats


def _write_report(report_path: str, report: Dict):
    if not report_path:
        return
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _build_report(
    per_split_stats: Dict[str, Dict],
    global_stats: Dict,
    timing: bool,
    include_strip: bool,
    both_mode_enabled: bool,
    variant_suffixes: List[str],
) -> Dict:
    report = {"splits": {}, "global": {}}
    for split, s in per_split_stats.items():
        report["splits"][split] = _format_stats(s, timing, include_strip)
    report["global"] = _format_stats(global_stats, timing, include_strip)
    report["both_mode_enabled"] = both_mode_enabled
    report["variant_suffixes"] = variant_suffixes
    return report


def _format_stats(stats: Dict, timing_enabled: bool, include_strip: bool) -> Dict:
    avg_tokenize_calls = stats["tokenize_calls"] / stats["total_seen"] if stats["total_seen"] else 0.0
    avg_prompt_tokens_before = (
        stats["prompt_tokens_sum_before"] / stats["total_seen"] if stats["total_seen"] else 0.0
    )
    avg_prompt_tokens_after = (
        stats["prompt_tokens_sum_after"] / max(1, stats["kept"] + stats["truncated"])
    )
    avg_completion_tokens = (
        stats["completion_tokens_sum"] / stats["total_seen"] if stats["total_seen"] else 0.0
    )

    formatted = {
        "originals_seen": stats["originals_seen"],
        "originals_emitted_2": stats["originals_emitted_2"],
        "originals_emitted_1": stats["originals_emitted_1"],
        "originals_emitted_0": stats["originals_emitted_0"],
        "variants_emitted_total": stats["variants_emitted_total"],
        "norag_emitted": stats["norag_emitted"],
        "rag_emitted": stats["rag_emitted"],
        "variant_emitted_counts": dict(stats["variant_emitted_counts"]),
        "total_seen": stats["total_seen"],
        "kept": stats["kept"],
        "truncated": stats["truncated"],
        "stripped": stats["stripped"],
        "dropped": stats["dropped"],
        "drop_reasons": dict(stats["drop_reasons"]),
        "drop_reasons_by_variant": {k: dict(v) for k, v in stats["drop_reasons_by_variant"].items()},
        "safeguards": stats["safeguards"],
        "tokenize_calls_total": stats["tokenize_calls"],
        "tokenize_calls_avg_per_example": avg_tokenize_calls,
        "prompt_tokens_avg_before": avg_prompt_tokens_before,
        "prompt_tokens_avg_after": avg_prompt_tokens_after,
        "prompt_tokens_max_before": stats["prompt_tokens_max_before"],
        "prompt_tokens_max_after": stats["prompt_tokens_max_after"],
        "prompt_tokens_over_model_max": stats["prompt_tokens_over_model_max"],
        "completion_tokens_avg": avg_completion_tokens,
        "completion_tokens_max": stats["completion_tokens_max"],
        "removed_prompt_tokens_total": stats["removed_prompt_tokens_total"],
        "examples_with_prompt_tokens_over_budget": stats["examples_with_prompt_tokens_over_budget"],
    }
    if timing_enabled:
        formatted["timing_seconds"] = dict(stats["timing"])
    if include_strip:
        formatted["file_sep_segments_removed"] = stats["file_sep_segments_removed"]
        formatted["file_sep_tokens_removed_estimate"] = stats["file_sep_tokens_removed_estimate"]
    return formatted


def main():
    args = _parse_args()
    if args.strip_report is None:
        args.strip_report = bool(args.strip_to_pure_fim or args.emit_both_rag_and_norag)
    args.max_seq_length = args.max_seq_length or args.max_length
    suffixes = [s.strip() for s in args.both_mode_suffixes.split(",") if s.strip()]
    if len(suffixes) != 2:
        raise ValueError("--both_mode_suffixes must contain exactly two comma-separated values.")
    args.both_mode_suffixes_list = suffixes

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

    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is not None and model_max_length > UNKNOWN_MODEL_MAX_LENGTH:
        model_max_length = None

    per_split_stats: Dict[str, Dict] = {}
    processed_splits = {}
    for split_name in ["train", "validation"]:
        timing = defaultdict(float) if args.time_breakdown else None
        token_helper = TokenLenHelper(
            tokenizer=tokenizer,
            cache_size=args.token_len_cache_size,
            max_calls_per_example=args.max_tokenize_calls_per_example,
            timing=timing,
        )
        processed, split_stats = _process_split(
            split_name,
            datasets[split_name],
            tokenizer,
            token_helper,
            args,
            model_max_length,
        )
        if timing is not None:
            split_stats["timing"].update(timing)
        per_split_stats[split_name] = split_stats
        processed_splits[split_name] = processed

    global_stats = _merge_global_stats(per_split_stats)

    _print_dataset_stats(stats, args.dataset_format)
    _log(f"train size: {len(processed_splits['train'])}", "normal", args.log_level)
    _log(f"val size: {len(processed_splits['validation'])}", "normal", args.log_level)

    if args.out_train:
        _write_jsonl(processed_splits["train"], args.out_train, args.encoding)
        _log(f"Wrote converted train split to {args.out_train}", "normal", args.log_level)
    if args.out_val:
        _write_jsonl(processed_splits["validation"], args.out_val, args.encoding)
        _log(f"Wrote converted val split to {args.out_val}", "normal", args.log_level)

    if args.report_path or args.truncate_report_path:
        report = _build_report(
            per_split_stats,
            global_stats,
            args.time_breakdown,
            args.strip_report,
            args.emit_both_rag_and_norag,
            args.both_mode_suffixes_list,
        )
        if args.truncate_report_path:
            _write_report(args.truncate_report_path, report)
        if args.report_path:
            _write_report(args.report_path, report)

    kept_total = global_stats["kept"] + global_stats["truncated"]
    dropped_total = global_stats["dropped"]
    summary = (
        f"Summary: seen={global_stats['total_seen']} kept={kept_total} "
        f"(unchanged/stripped={global_stats['kept']}, truncated={global_stats['truncated']}) "
        f"dropped={dropped_total} "
        f"avg_tokenize_calls={global_stats['tokenize_calls'] / max(1, global_stats['total_seen']):.2f}"
    )
    if args.emit_both_rag_and_norag:
        summary += (
            f" variants_emitted_total={global_stats['variants_emitted_total']} "
            f"originals={global_stats['originals_seen']} "
            f"2x/1x/0x=({global_stats['originals_emitted_2']}/"
            f"{global_stats['originals_emitted_1']}/{global_stats['originals_emitted_0']})"
        )
    _log(summary, "normal", args.log_level)
    if LOG_LEVELS[args.log_level] >= LOG_LEVELS["normal"]:
        _print_examples(processed_splits["train"], tokenizer, args.example_count, args.log_level)


if __name__ == "__main__":
    main()
