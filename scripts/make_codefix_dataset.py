import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rag_retrieval import BM25Index, tokenize

SKIP_DIRS = {".git", ".idea", "target", "build", "out", ".gradle", ".mvn"}
MUTATION_TYPES = ("swap_args", "missing_arg", "extra_arg")
CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "synchronized",
    "return",
    "throw",
    "new",
}
SYSTEM_PROMPT = (
    "You fix exactly one Java statement.\n"
    "Return exactly one fenced Java code block with one corrected statement line.\n"
    "Do not output explanations or any text outside the code block."
)
PROBLEM_BY_MUTATION = {
    "swap_args": "argument order mismatch in method invocation",
    "missing_arg": "actual and formal argument lists differ in length: missing argument",
    "extra_arg": "actual and formal argument lists differ in length: extra argument",
}
IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*$")


@dataclass(frozen=True)
class ParsedInvocation:
    line_text: str
    open_paren: int
    close_paren: int
    args: Tuple[str, ...]


@dataclass(frozen=True)
class CodefixCandidate:
    file: str
    line_index: int
    original_line: str
    broken_line: str
    mutation_type: str
    compiler_problems: Tuple[str, ...]

    @property
    def id(self) -> str:
        return f"{self.file}::{self.line_index}::{self.mutation_type}"


@dataclass(frozen=True)
class StatementEntry:
    file: str
    line_index: int
    text: str


def iter_java_files(repo: Path) -> Iterable[Path]:
    for path in repo.rglob("*.java"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def _scan_parentheses_pairs(text: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    stack: List[int] = []
    in_single = False
    in_double = False
    escape = False

    for idx, ch in enumerate(text):
        if in_single or in_double:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if in_single and ch == "'":
                in_single = False
                continue
            if in_double and ch == '"':
                in_double = False
                continue
            continue

        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            stack.append(idx)
            continue
        if ch == ")":
            if not stack:
                continue
            open_idx = stack.pop()
            pairs.append((open_idx, idx))
    return pairs


def _extract_method_name(text: str, open_paren: int) -> Optional[str]:
    idx = open_paren - 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    if idx < 0:
        return None
    if not (text[idx].isalnum() or text[idx] in "_$]>"):
        return None

    end = idx + 1
    while idx >= 0 and (text[idx].isalnum() or text[idx] in "_$.>"):
        idx -= 1
    token = text[idx + 1:end]
    if not token:
        return None
    method = token.split(".")[-1]
    method = method.split(">")[-1]
    if not method:
        return None
    if not IDENTIFIER_RE.fullmatch(method):
        return None
    if method in CONTROL_KEYWORDS:
        return None
    return method


def extract_invocation_args(arg_text: str) -> Optional[List[str]]:
    if not arg_text.strip():
        return []

    args_raw: List[str] = []
    start = 0
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_single = False
    in_double = False
    escape = False

    for idx, ch in enumerate(arg_text):
        if in_single or in_double:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if in_single and ch == "'":
                in_single = False
                continue
            if in_double and ch == '"':
                in_double = False
                continue
            continue

        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue

        if ch == "(":
            depth_paren += 1
            continue
        if ch == ")":
            if depth_paren <= 0:
                return None
            depth_paren -= 1
            continue
        if ch == "[":
            depth_bracket += 1
            continue
        if ch == "]":
            if depth_bracket <= 0:
                return None
            depth_bracket -= 1
            continue
        if ch == "{":
            depth_brace += 1
            continue
        if ch == "}":
            if depth_brace <= 0:
                return None
            depth_brace -= 1
            continue

        if ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            args_raw.append(arg_text[start:idx])
            start = idx + 1

    if in_single or in_double or depth_paren != 0 or depth_bracket != 0 or depth_brace != 0:
        return None

    args_raw.append(arg_text[start:])

    args: List[str] = []
    for raw in args_raw:
        cleaned = raw.strip()
        if not cleaned:
            return None
        args.append(cleaned)
    return args


def parse_invocation_statement(line: str) -> Optional[ParsedInvocation]:
    line_no_nl = line.rstrip("\r\n")
    if not line_no_nl:
        return None

    stripped = line_no_nl.strip()
    if not stripped or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
        return None

    right_trimmed = line_no_nl.rstrip()
    if not right_trimmed.endswith(";"):
        return None
    if "(" not in right_trimmed or ")" not in right_trimmed:
        return None

    semicolon_idx = len(right_trimmed) - 1
    pairs = _scan_parentheses_pairs(right_trimmed[:semicolon_idx])
    if not pairs:
        return None

    pairs.sort(key=lambda p: p[1], reverse=True)
    for open_idx, close_idx in pairs:
        if close_idx >= semicolon_idx:
            continue
        method_name = _extract_method_name(right_trimmed, open_idx)
        if method_name is None:
            continue

        args_text = right_trimmed[open_idx + 1: close_idx]
        args = extract_invocation_args(args_text)
        if args is None:
            continue
        return ParsedInvocation(
            line_text=right_trimmed,
            open_paren=open_idx,
            close_paren=close_idx,
            args=tuple(args),
        )

    return None


def _candidate_rng(seed: int, file: str, line_index: int, mutation_type: str) -> random.Random:
    raw = f"{seed}|{file}|{line_index}|{mutation_type}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def mutate_statement(parsed: ParsedInvocation, mutation_type: str, rng: random.Random) -> Optional[str]:
    args = list(parsed.args)
    if mutation_type == "swap_args":
        if len(args) < 2:
            return None
        pairs = [
            (i, j)
            for i in range(len(args))
            for j in range(i + 1, len(args))
            if args[i] != args[j]
        ]
        if not pairs:
            return None
        i, j = rng.choice(pairs)
        args[i], args[j] = args[j], args[i]
    elif mutation_type == "missing_arg":
        if len(args) < 1:
            return None
        drop_idx = rng.randrange(len(args))
        args = [arg for idx, arg in enumerate(args) if idx != drop_idx]
    elif mutation_type == "extra_arg":
        if len(args) < 1:
            return None
        duplicate_idx = rng.randrange(len(args))
        insert_idx = rng.randrange(len(args) + 1)
        args.insert(insert_idx, args[duplicate_idx])
    else:
        raise ValueError(f"Unsupported mutation_type={mutation_type}")

    new_inside = ", ".join(args)
    mutated = (
        parsed.line_text[: parsed.open_paren + 1]
        + new_inside
        + parsed.line_text[parsed.close_paren:]
    )
    if mutated == parsed.line_text:
        return None
    return mutated


def collect_candidates(repo: Path, seed: int) -> Tuple[Dict[str, List[CodefixCandidate]], List[StatementEntry]]:
    pools: Dict[str, List[CodefixCandidate]] = {m: [] for m in MUTATION_TYPES}
    statements: List[StatementEntry] = []

    files = sorted(iter_java_files(repo))
    for file_path in files:
        rel_file = file_path.relative_to(repo).as_posix()
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        for line_index, raw_line in enumerate(lines):
            parsed = parse_invocation_statement(raw_line)
            if parsed is None:
                continue

            statements.append(StatementEntry(file=rel_file, line_index=line_index, text=parsed.line_text))

            for mutation_type in MUTATION_TYPES:
                rng = _candidate_rng(seed, rel_file, line_index, mutation_type)
                broken = mutate_statement(parsed, mutation_type, rng)
                if broken is None:
                    continue
                pools[mutation_type].append(
                    CodefixCandidate(
                        file=rel_file,
                        line_index=line_index,
                        original_line=parsed.line_text,
                        broken_line=broken,
                        mutation_type=mutation_type,
                        compiler_problems=(PROBLEM_BY_MUTATION[mutation_type],),
                    )
                )

    return pools, statements


def target_counts_by_mutation(total_samples: int) -> Dict[str, int]:
    if total_samples < len(MUTATION_TYPES):
        raise ValueError(
            f"target_total must be >= {len(MUTATION_TYPES)} to support all mutation types"
        )
    base = total_samples // len(MUTATION_TYPES)
    remainder = total_samples % len(MUTATION_TYPES)
    out: Dict[str, int] = {}
    for idx, mutation_type in enumerate(MUTATION_TYPES):
        out[mutation_type] = base + (1 if idx < remainder else 0)
    return out


def _find_file_subset_with_exact_count(file_counts: Dict[str, int], target: int) -> Optional[Set[str]]:
    if target == 0:
        return set()

    items = sorted(file_counts.items(), key=lambda kv: (kv[1], kv[0]))
    dp: Dict[int, Tuple[Optional[int], Optional[str]]] = {0: (None, None)}

    for file_name, count in items:
        if count <= 0 or count > target:
            continue
        current_sums = sorted(dp.keys(), reverse=True)
        for prev_sum in current_sums:
            next_sum = prev_sum + count
            if next_sum > target or next_sum in dp:
                continue
            dp[next_sum] = (prev_sum, file_name)

    if target not in dp:
        return None

    chosen: Set[str] = set()
    cursor = target
    while cursor != 0:
        prev_sum, file_name = dp[cursor]
        if prev_sum is None or file_name is None:
            break
        chosen.add(file_name)
        cursor = prev_sum
    return chosen


def choose_candidates_with_split(
    pools: Dict[str, List[CodefixCandidate]],
    targets: Dict[str, int],
    total_samples: int,
    val_count: int,
    seed: int,
    max_attempts: int,
) -> Tuple[List[CodefixCandidate], List[CodefixCandidate]]:
    for mutation_type, required in targets.items():
        available = len(pools.get(mutation_type, []))
        if available < required:
            raise RuntimeError(
                f"Not enough candidates for {mutation_type}: required={required}, available={available}"
            )

    rng = random.Random(seed)
    for _ in range(max_attempts):
        selected: List[CodefixCandidate] = []
        for mutation_type in MUTATION_TYPES:
            pool = list(pools[mutation_type])
            rng.shuffle(pool)
            selected.extend(pool[: targets[mutation_type]])

        if len(selected) != total_samples:
            raise RuntimeError(
                f"Internal selection error: expected {total_samples}, got {len(selected)}"
            )

        file_counts = Counter(candidate.file for candidate in selected)
        val_files = _find_file_subset_with_exact_count(dict(file_counts), val_count)
        if val_files is None:
            continue

        val = [c for c in selected if c.file in val_files]
        train = [c for c in selected if c.file not in val_files]
        if len(val) != val_count:
            continue
        if len(train) != total_samples - val_count:
            continue
        return train, val

    raise RuntimeError(
        "Failed to find a split-by-file selection with exact val_count after "
        f"{max_attempts} attempts. Try increasing candidate diversity, changing seed, "
        "or reducing target_total."
    )


class StatementBM25Retriever:
    def __init__(
        self,
        statements: Sequence[StatementEntry],
        *,
        same_file_window: int,
        drop_stopwords: bool,
        use_identifiers_only: bool,
    ) -> None:
        self.statements = list(statements)
        self.same_file_window = same_file_window
        self.drop_stopwords = drop_stopwords
        self.use_identifiers_only = use_identifiers_only
        self.index = BM25Index()
        docs = [
            tokenize(
                statement.text,
                drop_stopwords=self.drop_stopwords,
                identifiers_only=self.use_identifiers_only,
            )
            for statement in self.statements
        ]
        self.index.add_documents(docs)

    def retrieve(self, candidate: CodefixCandidate, top_k: int) -> List[str]:
        query_tokens = tokenize(
            candidate.broken_line,
            drop_stopwords=self.drop_stopwords,
            identifiers_only=self.use_identifiers_only,
        )
        ranked = self.index.search(query_tokens, top_k=max(top_k * 20, 80))

        out: List[str] = []
        seen: Set[str] = set()

        for doc_id, _score in ranked:
            statement = self.statements[doc_id]
            if statement.file == candidate.file and abs(statement.line_index - candidate.line_index) <= self.same_file_window:
                continue
            text = statement.text.rstrip("\r\n")
            if text == candidate.original_line or text == candidate.broken_line:
                continue
            if text in seen:
                continue
            out.append(text)
            seen.add(text)
            if len(out) >= top_k:
                break

        if len(out) < top_k:
            for statement in self.statements:
                if statement.file == candidate.file and abs(statement.line_index - candidate.line_index) <= self.same_file_window:
                    continue
                text = statement.text.rstrip("\r\n")
                if text == candidate.original_line or text == candidate.broken_line:
                    continue
                if text in seen:
                    continue
                out.append(text)
                seen.add(text)
                if len(out) >= top_k:
                    break

        return out


def _format_user_prompt(augmentations: Sequence[str], problems: Sequence[str], broken_line: str) -> str:
    if augmentations:
        aug_text = "\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(augmentations))
    else:
        aug_text = "1. (no similar statements found)"

    problem_text = "\n".join(f"{idx + 1}. {problem}" for idx, problem in enumerate(problems))

    return (
        "With these similar correct statements:\n"
        f"{aug_text}\n\n"
        "Fix this problems:\n"
        f"{problem_text}\n\n"
        "In this code snippet to fix:\n"
        "```java\n"
        f"{broken_line}\n"
        "```\n\n"
        "Answer only one Java code block with the fixed replacement statement."
    )


def _format_chat_prompt(messages: Sequence[Dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        chunks.append(f"<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def build_record(candidate: CodefixCandidate, augmentations: Sequence[str]) -> dict:
    broken_line = candidate.broken_line.rstrip("\r\n")
    original_line = candidate.original_line.rstrip("\r\n")
    problems = list(candidate.compiler_problems)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _format_user_prompt(augmentations=augmentations, problems=problems, broken_line=broken_line),
        },
    ]

    prompt = _format_chat_prompt(messages)
    completion = f"```java\n{original_line}\n```"

    return {
        "id": candidate.id,
        "file": candidate.file,
        "line_index": candidate.line_index,
        "mutation_type": candidate.mutation_type,
        "original_line": original_line,
        "broken_line": broken_line,
        "compiler_problems": problems,
        "augmentations": list(augmentations),
        "messages": messages,
        "prompt": prompt,
        "completion": completion,
    }


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic Java code-fix dataset for chat-completions training.")
    parser.add_argument("--repo", required=True, help="Path to Java repository")
    parser.add_argument("--out_train", default="data/codefix_train.jsonl")
    parser.add_argument("--out_val", default="data/codefix_val.jsonl")
    parser.add_argument("--target_total", type=int, default=5000)
    parser.add_argument("--val_count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bm25_top_k", type=int, default=4)
    parser.add_argument("--bm25_same_file_window", type=int, default=80)
    parser.add_argument("--bm25_drop_stopwords", action="store_true", default=True)
    parser.add_argument("--no-bm25_drop_stopwords", dest="bm25_drop_stopwords", action="store_false")
    parser.add_argument("--bm25_use_identifiers_only", action="store_true")
    parser.add_argument("--selection_attempts", type=int, default=300)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_total <= 0:
        raise ValueError("--target_total must be > 0")
    if args.val_count < 0:
        raise ValueError("--val_count must be >= 0")
    if args.val_count >= args.target_total:
        raise ValueError("--val_count must be smaller than --target_total")
    if args.bm25_top_k <= 0:
        raise ValueError("--bm25_top_k must be > 0")
    if args.selection_attempts <= 0:
        raise ValueError("--selection_attempts must be > 0")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    repo = Path(args.repo).resolve()
    if not repo.exists():
        raise SystemExit(f"Repository path does not exist: {repo}")

    pools, statements = collect_candidates(repo=repo, seed=args.seed)
    if not statements:
        raise SystemExit("No statement-level invocation opportunities found in repository.")

    targets = target_counts_by_mutation(args.target_total)
    train_candidates, val_candidates = choose_candidates_with_split(
        pools=pools,
        targets=targets,
        total_samples=args.target_total,
        val_count=args.val_count,
        seed=args.seed,
        max_attempts=args.selection_attempts,
    )

    retriever = StatementBM25Retriever(
        statements=statements,
        same_file_window=args.bm25_same_file_window,
        drop_stopwords=args.bm25_drop_stopwords,
        use_identifiers_only=args.bm25_use_identifiers_only,
    )

    train_rows = [
        build_record(candidate, retriever.retrieve(candidate, top_k=args.bm25_top_k))
        for candidate in train_candidates
    ]
    val_rows = [
        build_record(candidate, retriever.retrieve(candidate, top_k=args.bm25_top_k))
        for candidate in val_candidates
    ]

    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_val, val_rows)

    total_rows = len(train_rows) + len(val_rows)
    mutation_counts = Counter(row["mutation_type"] for row in train_rows + val_rows)
    train_mutation_counts = Counter(row["mutation_type"] for row in train_rows)
    val_mutation_counts = Counter(row["mutation_type"] for row in val_rows)

    print("[codefix] generation complete")
    print(f"[codefix] train={len(train_rows)} val={len(val_rows)} total={total_rows}")
    print(f"[codefix] mutation_targets={dict(targets)}")
    print(f"[codefix] mutation_counts_total={dict(mutation_counts)}")
    print(f"[codefix] mutation_counts_train={dict(train_mutation_counts)}")
    print(f"[codefix] mutation_counts_val={dict(val_mutation_counts)}")
    print(f"[codefix] out_train={out_train}")
    print(f"[codefix] out_val={out_val}")


if __name__ == "__main__":
    main()
