import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rag_retrieval

SKIP_DIRS = {".git", ".idea", "target", "build", "out", ".gradle", ".mvn"}
PROMPT_CONTEXT_LINES = 10
FUZZY_THRESHOLD = 0.9
FUZZY_MARGIN = 0.05


@dataclass
class BenchRecord:
    task_id: str
    prompt: str
    ground_truth: str
    fpath_tuple: Sequence[str]
    line_no: Optional[int]
    metadata: dict


@dataclass
class MatchResult:
    status: str
    resolved_path: Optional[str]
    line_index: Optional[int]
    strategy: Optional[str]
    confidence: int
    reason: Optional[str] = None


def iter_java_files(repo: Path) -> Iterable[Path]:
    for p in repo.rglob("*.java"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def extract_header(lines: Sequence[str]) -> Tuple[str, int]:
    i = 0
    header = []
    if i < len(lines) and lines[i].lstrip().startswith("package "):
        header.append(lines[i])
        i += 1
        while i < len(lines) and lines[i].strip() == "":
            header.append(lines[i])
            i += 1
    saw_import = False
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith("import "):
            saw_import = True
            header.append(lines[i])
            i += 1
            continue
        if saw_import and lines[i].strip() == "":
            header.append(lines[i])
            i += 1
            continue
        break
    return ("".join(header), i)


def good_target_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) < 4:
        return False
    if re.fullmatch(r"[{}();]+", s):
        return False
    if s.startswith("package ") or s.startswith("import "):
        return False
    return True


def normalize_ground_truth(gt_raw: str):
    gt_nl_stripped = gt_raw.rstrip("\r\n")
    gt_trimmed = gt_nl_stripped.strip()
    gt_ws_norm = " ".join(gt_nl_stripped.split())
    return {
        "gt_raw": gt_raw,
        "gt_nl_stripped": gt_nl_stripped,
        "gt_trimmed": gt_trimmed,
        "gt_ws_norm": gt_ws_norm,
    }


def line_fingerprints(line: str) -> Set[str]:
    base = line.rstrip("\r\n")
    return {base, " ".join(base.split())}


def _line_equals(a: str, b: str) -> bool:
    return a.rstrip("\r\n") == b.rstrip("\r\n")


def _best_path_suffix_score(candidate: Path, tuple_parts: Sequence[str]) -> Tuple[int, int, str]:
    candidate_parts = list(candidate.parts)
    tuple_parts_list = list(tuple_parts)
    score = 0
    while score < min(len(candidate_parts), len(tuple_parts_list)):
        if candidate_parts[-(score + 1)] == tuple_parts_list[-(score + 1)]:
            score += 1
        else:
            break
    return (-score, len(candidate_parts), candidate.as_posix())


def build_filename_index(paths: Sequence[Path], repo: Path) -> Dict[str, List[Path]]:
    by_name: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        name = p.name
        rel = p.relative_to(repo)
        by_name[name].append(rel)
    for name in by_name:
        by_name[name].sort()
    return by_name


def load_bench_records(paths: Sequence[Path]) -> List[BenchRecord]:
    records: List[BenchRecord] = []
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                for line_no, raw in enumerate(f, 1):
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"[bench] skip malformed JSON in {p} line {line_no}")
                        continue
                    meta = obj.get("metadata") or {}
                    gt = meta.get("ground_truth")
                    fpath_tuple = meta.get("fpath_tuple")
                    line_idx = meta.get("line_no")
                    prompt = obj.get("prompt", "")
                    task_id = meta.get("task_id", f"{p.name}:{line_no}")
                    if gt is None or fpath_tuple is None:
                        print(f"[bench] missing ground_truth or fpath_tuple in {p} line {line_no}")
                        continue
                    if isinstance(fpath_tuple, str):
                        fpath_tuple = [fpath_tuple]
                    records.append(BenchRecord(
                        task_id=task_id,
                        prompt=prompt,
                        ground_truth=gt,
                        fpath_tuple=list(fpath_tuple),
                        line_no=int(line_idx) if isinstance(line_idx, (int, float, str)) and str(line_idx).lstrip("+-").isdigit() else None,
                        metadata=meta,
                    ))
        except FileNotFoundError:
            print(f"[bench] missing benchmark file {p}")
    return records


def gather_bench_paths(bench_jsonl: Sequence[str], bench_dir: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    for p in bench_jsonl:
        paths.append(Path(p))
    if bench_dir:
        for candidate in Path(bench_dir).glob("*.jsonl"):
            paths.append(candidate)
    seen = {}
    uniq_paths: List[Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            uniq_paths.append(resolved)
            seen[resolved] = True
    return sorted(uniq_paths)


def get_lines_cache(path: Path, cache: Dict[Path, List[str]]) -> List[str]:
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    return cache[path]


def resolve_candidate_paths(record: BenchRecord, repo: Path, filename_index: Dict[str, List[Path]]) -> List[Path]:
    tuple_parts = list(record.fpath_tuple)
    candidate_paths: List[Path] = []
    for i in range(len(tuple_parts) + 1):
        suffix = Path(*tuple_parts[i:])
        if suffix.suffix != ".java":
            continue
        candidate = repo / suffix
        if candidate.exists():
            candidate_paths.append(suffix)
    if candidate_paths:
        return candidate_paths
    fname = Path(*tuple_parts).name
    candidates = filename_index.get(fname, [])
    candidates = sorted(candidates, key=lambda p: _best_path_suffix_score(p, tuple_parts))
    return candidates


def _match_using_indices(lines: Sequence[str], idx: int, gt_raw: str, gt_nl_stripped: str) -> Optional[MatchResult]:
    if idx < 0 or idx >= len(lines):
        return None
    if lines[idx] == gt_raw or _line_equals(lines[idx], gt_raw) or _line_equals(lines[idx], gt_nl_stripped):
        return MatchResult(status="matched", resolved_path=None, line_index=idx, strategy="line_no_exact", confidence=100)
    return None


def _window_search(lines: Sequence[str], base_idx: int, gt_nl_stripped: str, window: int) -> Optional[int]:
    matches: List[int] = []
    start = max(0, base_idx - window)
    end = min(len(lines), base_idx + window + 1)
    for i in range(start, end):
        if _line_equals(lines[i], gt_nl_stripped):
            matches.append(i)
    if not matches:
        return None
    matches.sort(key=lambda i: (abs(i - base_idx), i))
    return matches[0]


def _global_exact_search(lines: Sequence[str], gt_nl_stripped: str) -> List[int]:
    return [i for i, line in enumerate(lines) if _line_equals(line, gt_nl_stripped)]


def _disambiguate_with_prompt(lines: Sequence[str], candidates: List[int], prompt: str) -> Optional[int]:
    prompt_tail = "\n".join(prompt.splitlines()[-PROMPT_CONTEXT_LINES:])
    if not prompt_tail.strip():
        return None
    scores = []
    for idx in candidates:
        context = "".join(lines[max(0, idx - PROMPT_CONTEXT_LINES): min(len(lines), idx + PROMPT_CONTEXT_LINES)])
        ratio = SequenceMatcher(None, prompt_tail, context).ratio()
        scores.append((ratio, idx))
    scores.sort(reverse=True)
    if len(scores) == 1:
        return scores[0][1]
    if scores[0][0] - scores[1][0] > 0.05:
        return scores[0][1]
    return None


def _fuzzy_search(lines: Sequence[str], gt_ws_norm: str) -> Optional[int]:
    best = None
    runner_up = None
    for idx, line in enumerate(lines):
        cand = " ".join(line.rstrip("\r\n").split())
        if not cand:
            continue
        ratio = SequenceMatcher(None, gt_ws_norm, cand).ratio()
        if ratio < FUZZY_THRESHOLD:
            continue
        if best is None or ratio > best[0]:
            runner_up = best
            best = (ratio, idx)
        elif runner_up is None or ratio > runner_up[0]:
            runner_up = (ratio, idx)
    if best is None:
        return None
    if runner_up and (best[0] - runner_up[0]) < FUZZY_MARGIN:
        return -1
    return best[1]


def match_record(record: BenchRecord, repo: Path, filename_index: Dict[str, List[Path]], lines_cache: Dict[Path, List[str]], match_window: int, enable_fuzzy: bool) -> MatchResult:
    normalized = normalize_ground_truth(record.ground_truth)
    candidate_paths = resolve_candidate_paths(record, repo, filename_index)
    if not candidate_paths:
        return MatchResult(status="unmatched_file", resolved_path=None, line_index=None, strategy=None, confidence=0, reason="no_candidate_file")
    best_match: Optional[MatchResult] = None
    for rel_path in candidate_paths:
        abs_path = repo / rel_path
        try:
            lines = get_lines_cache(abs_path, lines_cache)
        except FileNotFoundError:
            continue
        base_indices = []
        if record.line_no is not None:
            base_indices.extend([record.line_no, record.line_no - 1])
        seen_idx = set()
        for idx in base_indices:
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            res = _match_using_indices(lines, idx, normalized["gt_raw"], normalized["gt_nl_stripped"])
            if res:
                res.resolved_path = rel_path.as_posix()
                res.strategy = "line_no_exact_0based" if idx == record.line_no else "line_no_exact_1based"
                return res
            win_idx = _window_search(lines, idx, normalized["gt_nl_stripped"], match_window)
            if win_idx is not None:
                return MatchResult(status="matched", resolved_path=rel_path.as_posix(), line_index=win_idx, strategy="window_search", confidence=95)
        global_hits = _global_exact_search(lines, normalized["gt_nl_stripped"])
        if len(global_hits) == 1:
            return MatchResult(status="matched", resolved_path=rel_path.as_posix(), line_index=global_hits[0], strategy="global_exact", confidence=90)
        if len(global_hits) > 1:
            disamb = _disambiguate_with_prompt(lines, global_hits, record.prompt)
            if disamb is not None:
                return MatchResult(status="matched", resolved_path=rel_path.as_posix(), line_index=disamb, strategy="prompt_disambiguation", confidence=80)
            best_match = MatchResult(status="unmatched_ambiguous", resolved_path=rel_path.as_posix(), line_index=None, strategy="global_exact", confidence=30, reason="multiple_matches")
        if enable_fuzzy:
            fuzzy_idx = _fuzzy_search(lines, normalized["gt_ws_norm"])
            if fuzzy_idx is not None:
                if fuzzy_idx >= 0:
                    return MatchResult(status="matched", resolved_path=rel_path.as_posix(), line_index=fuzzy_idx, strategy="fuzzy", confidence=60)
                best_match = MatchResult(status="unmatched_ambiguous", resolved_path=rel_path.as_posix(), line_index=None, strategy="fuzzy", confidence=30, reason="ambiguous_fuzzy")
    if best_match:
        return best_match
    return MatchResult(status="unmatched_line", resolved_path=candidate_paths[0].as_posix(), line_index=None, strategy=None, confidence=0, reason="no_line_match")


def build_samples_from_file(
        path: Path,
        repo: Path,
        max_prefix_lines: int,
        max_suffix_lines: int,
        include_header: bool,
        max_samples_per_file: int,
        rng: random.Random,
        excluded_by_file: Optional[Dict[str, Set[int]]] = None,
        text_blocklist_by_file: Optional[Dict[str, Set[str]]] = None,
        global_text_blocklist: Optional[Set[str]] = None,
        exclusion_counter: Optional[Dict[str, int]] = None,
        rag_retriever: Optional[rag_retrieval.RagRetriever] = None,
        rag_insert_location: str = "prefix_head",
):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)

    header_text, header_end = extract_header(lines) if include_header else ("", 0)

    rel_path = path.relative_to(repo).as_posix()
    excluded_indices = excluded_by_file.get(rel_path, set()) if excluded_by_file else set()
    file_text_blocklist = text_blocklist_by_file.get(rel_path, set()) if text_blocklist_by_file else set()

    candidates = []
    for i in range(header_end, len(lines)):
        if not good_target_line(lines[i]):
            continue
        if i in excluded_indices:
            if exclusion_counter is not None:
                exclusion_counter["line_match"] += 1
            continue
        fingerprints = line_fingerprints(lines[i])
        if file_text_blocklist and fingerprints & file_text_blocklist:
            if exclusion_counter is not None:
                exclusion_counter["text_block_file"] += 1
            continue
        if global_text_blocklist and fingerprints & global_text_blocklist:
            if exclusion_counter is not None:
                exclusion_counter["text_block_global"] += 1
            continue
        candidates.append(i)
    if not candidates:
        return []

    rng.shuffle(candidates)
    candidates = candidates[:max_samples_per_file]

    samples = []
    for i in candidates:
        target = lines[i]
        before = lines[max(header_end, i - max_prefix_lines): i]
        after = lines[i + 1: i + 1 + max_suffix_lines]

        prefix = header_text + "".join(before)
        suffix = "".join(after)

        rag_meta = {}
        if rag_retriever:
            rag_block, rag_list, rag_query = rag_retriever.retrieve(rel_path, i, before, after, target)
            if rag_block:
                if rag_insert_location == "prefix_tail":
                    prefix = prefix + rag_block
                else:
                    prefix = rag_block + prefix
                rag_meta = {
                    "rag": rag_list,
                    "rag_query": rag_query,
                    "rag_k_used": len(rag_list),
                }

        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        completion = target

        sample = {
            "id": f"{path.as_posix()}::{i}",
            "file": path.as_posix(),
            "line_index": i,
            "prompt": prompt,
            "completion": completion,
        }
        if rag_meta:
            sample.update(rag_meta)
        samples.append(sample)
    return samples


def build_bench_masks(records: List[BenchRecord], repo: Path, filename_index: Dict[str, List[Path]], match_window: int, enable_fuzzy: bool, global_text_blocklist: bool):
    lines_cache: Dict[Path, List[str]] = {}
    excluded_by_file: Dict[str, Set[int]] = defaultdict(set)
    text_blocklist_by_file: Dict[str, Set[str]] = defaultdict(set)
    global_blocklist: Set[str] = set()
    matched_items = []
    unmatched_items = []
    referenced_files: Set[str] = set()

    for rec in records:
        match_res = match_record(rec, repo, filename_index, lines_cache, match_window, enable_fuzzy)
        normalized = normalize_ground_truth(rec.ground_truth)
        if match_res.resolved_path:
            referenced_files.add(match_res.resolved_path)
        if match_res.status == "matched" and match_res.resolved_path is not None and match_res.line_index is not None:
            excluded_by_file[match_res.resolved_path].add(match_res.line_index)
            text_blocklist_by_file[match_res.resolved_path].update(line_fingerprints(normalized["gt_nl_stripped"]))
            matched_items.append({
                "task_id": rec.task_id,
                "file": match_res.resolved_path,
                "line_index": match_res.line_index,
                "strategy": match_res.strategy,
                "confidence": match_res.confidence,
            })
        else:
            unmatched_items.append({
                "task_id": rec.task_id,
                "status": match_res.status,
                "reason": match_res.reason,
                "file": match_res.resolved_path,
            })
        if global_text_blocklist:
            global_blocklist.update(line_fingerprints(normalized["gt_nl_stripped"]))

    report = {
        "total_records": len(records),
        "matched": len(matched_items),
        "unmatched": len(unmatched_items),
        "matched_items": matched_items,
        "unmatched_items": unmatched_items,
        "referenced_files": sorted(referenced_files),
    }
    return excluded_by_file, text_blocklist_by_file, global_blocklist, referenced_files, report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to cloned repo")
    ap.add_argument("--out_train", default="data/train.jsonl")
    ap.add_argument("--out_val", default="data/val.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_prefix_lines", type=int, default=80)
    ap.add_argument("--max_suffix_lines", type=int, default=80)
    ap.add_argument("--include_header", action="store_true", help="Keep package+imports in prefix")

    ap.add_argument("--max_samples_per_file", type=int, default=60)
    ap.add_argument("--split_by_file", action="store_true", help="Val split by files (avoid leakage)")

    ap.add_argument("--bench_jsonl", action="append", default=[], help="Benchmark JSONL file (repeatable)")
    ap.add_argument("--bench_dir", help="Directory with benchmark JSONL files")
    ap.add_argument("--bench_match_window", type=int, default=60, help="Window for nearby line matches")
    ap.add_argument("--bench_fuzzy", action="store_true", help="Enable fuzzy benchmark matching")
    ap.add_argument("--bench_report_path", default="data/bench_mask_report.json")
    ap.add_argument("--bench_global_text_blocklist", action="store_true", default=True, help="Block any line text that appears in benchmark samples")
    ap.add_argument("--no-bench_global_text_blocklist", action="store_false", dest="bench_global_text_blocklist", help="Disable global text blocklisting from benchmark lines")
    ap.add_argument("--leak_policy", choices=["exclude_only", "downweight_files"], default="exclude_only")
    ap.add_argument("--bench_downweight_factor", type=float, default=0.25, help="Sampling multiplier for benchmark-referenced files when leak_policy=downweight_files")
    ap.add_argument("--rag_enable", action="store_true", help="Enable retrieval-augmented context insertion")
    ap.add_argument("--rag_k", type=int, default=4, help="Number of retrieved snippets per sample")
    ap.add_argument("--rag_max_chars", type=int, default=2000, help="Character budget for retrieval section")
    ap.add_argument("--rag_max_snippet_chars", type=int, default=600, help="Per-snippet character cap")
    ap.add_argument("--rag_method", choices=["bm25"], default="bm25")
    ap.add_argument("--rag_chunker", choices=["lines", "treesitter"], default="lines")
    ap.add_argument("--rag_chunk_lines", type=int, default=30)
    ap.add_argument("--rag_chunk_overlap", type=int, default=10)
    ap.add_argument("--rag_query_mode", choices=["masked_window", "ast_symbols", "hybrid"], default="hybrid")
    ap.add_argument("--rag_query_window_lines", type=int, default=20)
    ap.add_argument("--rag_drop_stopwords", action="store_true", default=True)
    ap.add_argument("--no-rag_drop_stopwords", dest="rag_drop_stopwords", action="store_false")
    ap.add_argument("--rag_use_identifiers_only", action="store_true", help="Keep only identifier-like tokens when building queries")
    ap.add_argument("--rag_exclude_same_file_window", type=int, default=80, help="Do not retrieve from same file within this line window")
    ap.add_argument("--rag_exclude_bench_targets", action="store_true", default=True, help="Filter retrieval that overlaps benchmark-masked lines")
    ap.add_argument("--no-rag_exclude_bench_targets", dest="rag_exclude_bench_targets", action="store_false")
    ap.add_argument("--rag_exclude_completion_text", action="store_true", default=True, help="Filter retrieval containing the completion line text")
    ap.add_argument("--no-rag_exclude_completion_text", dest="rag_exclude_completion_text", action="store_false")
    ap.add_argument("--rag_insert_location", choices=["prefix_head", "prefix_tail"], default="prefix_head")
    ap.add_argument("--rag_format", choices=["comment_block"], default="comment_block")
    ap.add_argument("--rag_cache_dir", help="Optional cache directory for retrieval index")
    return ap.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    rng = random.Random(args.seed)

    files = list(iter_java_files(repo))
    files.sort()
    if not files:
        raise SystemExit("No .java files found")

    filename_index = build_filename_index(files, repo)

    bench_paths = gather_bench_paths(args.bench_jsonl, args.bench_dir)
    if bench_paths:
        bench_records = load_bench_records(bench_paths)
        excluded_by_file, text_blocklist_by_file, global_blocklist, referenced_files, bench_report = build_bench_masks(
            bench_records, repo, filename_index, args.bench_match_window, args.bench_fuzzy, args.bench_global_text_blocklist)
    else:
        bench_records = []
        excluded_by_file = {}
        text_blocklist_by_file = {}
        global_blocklist = set()
        referenced_files = set()
        bench_report = None

    retriever = None
    if args.rag_enable:
        if args.rag_format != "comment_block":
            raise SystemExit("Only comment_block RAG format is supported")
        retriever = rag_retrieval.RagRetriever(
            repo=repo,
            files=files,
            k=args.rag_k,
            max_chars=args.rag_max_chars,
            max_snippet_chars=args.rag_max_snippet_chars,
            method=args.rag_method,
            chunker=args.rag_chunker,
            chunk_lines=args.rag_chunk_lines,
            chunk_overlap=args.rag_chunk_overlap,
            query_mode=args.rag_query_mode,
            query_window_lines=args.rag_query_window_lines,
            drop_stopwords=args.rag_drop_stopwords,
            use_identifiers_only=args.rag_use_identifiers_only,
            exclude_same_file_window=args.rag_exclude_same_file_window,
            exclude_completion_text=args.rag_exclude_completion_text,
            exclude_bench_targets=args.rag_exclude_bench_targets,
            excluded_by_file=excluded_by_file if args.rag_exclude_bench_targets else {},
            text_blocklist_by_file=text_blocklist_by_file if args.rag_exclude_bench_targets else {},
            global_text_blocklist=global_blocklist if (args.rag_exclude_bench_targets and args.bench_global_text_blocklist) else set(),
        )
        if args.rag_cache_dir:
            print("[rag] cache_dir provided but persistence is not implemented; using in-memory index")

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val).parent.mkdir(parents=True, exist_ok=True)

    if args.split_by_file:
        rng.shuffle(files)
        n_val = max(1, int(len(files) * args.val_ratio))
        val_files = set(files[:n_val])
    else:
        val_files = set()

    exclusion_counter = defaultdict(int)
    per_bench_file_sample_count = defaultdict(int)

    train_f = open(args.out_train, "w", encoding="utf-8")
    val_f = open(args.out_val, "w", encoding="utf-8")

    n_train = n_val = 0
    for fpath in files:
        rel_path = fpath.relative_to(repo).as_posix()
        effective_max_samples = args.max_samples_per_file
        if args.leak_policy == "downweight_files" and rel_path in referenced_files:
            effective_max_samples = max(1, int(args.max_samples_per_file * args.bench_downweight_factor))
        samples = build_samples_from_file(
            fpath, repo, args.max_prefix_lines, args.max_suffix_lines,
            args.include_header, effective_max_samples, rng,
            excluded_by_file=excluded_by_file,
            text_blocklist_by_file=text_blocklist_by_file,
            global_text_blocklist=global_blocklist if args.bench_global_text_blocklist else None,
            exclusion_counter=exclusion_counter,
            rag_retriever=retriever if args.rag_enable else None,
            rag_insert_location=args.rag_insert_location,
        )
        if not samples:
            continue
        if rel_path in referenced_files:
            per_bench_file_sample_count[rel_path] += len(samples)

        if args.split_by_file:
            out = val_f if fpath in val_files else train_f
            for ex in samples:
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if fpath in val_files:
                n_val += len(samples)
            else:
                n_train += len(samples)
        else:
            for ex in samples:
                out = val_f if rng.random() < args.val_ratio else train_f
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                if out is val_f:
                    n_val += 1
                else:
                    n_train += 1

    train_f.close()
    val_f.close()

    rag_report_data = retriever.report() if retriever else None
    if rag_report_data:
        print(f"[rag] exclusion counts: {json.dumps(rag_report_data, sort_keys=True)}")

    if bench_paths:
        bench_report = bench_report or {}
        bench_report.update({
            "excluded_line_hits": dict(exclusion_counter),
            "bench_file_sample_counts": dict(per_bench_file_sample_count),
            "bench_paths": [p.as_posix() for p in bench_paths],
        })
        if rag_report_data:
            bench_report["rag_exclusions"] = rag_report_data
        report_path = Path(args.bench_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(bench_report, indent=2), encoding="utf-8")
        print(f"[bench] report written to {report_path}")

    print(f"Done. train={n_train}, val={n_val}")

    if bench_paths:
        overlaps_found = 0
        bench_line_set = {(fp, idx) for fp, idxs in excluded_by_file.items() for idx in idxs}
        try:
            with open(args.out_train, "r", encoding="utf-8") as f:
                for raw in f:
                    obj = json.loads(raw)
                    rel = str(Path(obj["file"]).resolve().relative_to(repo))
                    if (rel, obj.get("line_index")) in bench_line_set:
                        overlaps_found += 1
        except Exception as exc:  # pragma: no cover
            print(f"[bench] self-check skipped due to error: {exc}")
        if overlaps_found:
            print(f"[bench] WARNING: found {overlaps_found} overlapping samples in output")


if __name__ == "__main__":
    main()
