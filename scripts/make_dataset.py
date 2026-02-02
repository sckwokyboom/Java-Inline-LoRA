# scripts/make_dataset.py
import argparse, json, os, random, re
from pathlib import Path
from typing import Iterable, Tuple

SKIP_DIRS = {".git", ".idea", "target", "build", "out", ".gradle", ".mvn"}

def iter_java_files(repo: Path) -> Iterable[Path]:
    for p in repo.rglob("*.java"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p

def extract_header(lines) -> Tuple[str, int]:
    """
    Возвращает (header_text, header_end_index_exclusive)
    Header = package + непрерывный блок import (+ пустые строки между ними допускаем умеренно).
    """
    i = 0
    header = []
    # package
    if i < len(lines) and lines[i].lstrip().startswith("package "):
        header.append(lines[i])
        i += 1
        # пропустим пустые строки сразу после package
        while i < len(lines) and lines[i].strip() == "":
            header.append(lines[i])
            i += 1
    # imports block
    saw_import = False
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith("import "):
            saw_import = True
            header.append(lines[i]); i += 1
            continue
        if saw_import and lines[i].strip() == "":
            # пустые строки внутри import-блока
            header.append(lines[i]); i += 1
            continue
        break
    return ("".join(header), i)

def good_target_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # слишком короткие / чисто скобки — обычно неинтересно
    if len(s) < 4:
        return False
    if re.fullmatch(r"[{}();]+", s):
        return False
    # можно отфильтровать package/import строки как таргет (опционально)
    if s.startswith("package ") or s.startswith("import "):
        return False
    return True

def build_samples_from_file(
    path: Path,
    max_prefix_lines: int,
    max_suffix_lines: int,
    include_header: bool,
    max_samples_per_file: int,
    rng: random.Random,
):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)

    header_text, header_end = extract_header(lines) if include_header else ("", 0)

    # кандидаты: строки после header_end
    candidates = [i for i in range(header_end, len(lines)) if good_target_line(lines[i])]
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

        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        completion = target  # одна строка с \n, как в исходнике (keepends=True)

        samples.append({
            "id": f"{path.as_posix()}::{i}",
            "file": path.as_posix(),
            "line_index": i,
            "prompt": prompt,
            "completion": completion,
        })
    return samples

def main():
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

    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    rng = random.Random(args.seed)

    files = list(iter_java_files(repo))
    files.sort()
    if not files:
        raise SystemExit("No .java files found")

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val).parent.mkdir(parents=True, exist_ok=True)

    if args.split_by_file:
        rng.shuffle(files)
        n_val = max(1, int(len(files) * args.val_ratio))
        val_files = set(files[:n_val])
    else:
        val_files = set()

    train_f = open(args.out_train, "w", encoding="utf-8")
    val_f = open(args.out_val, "w", encoding="utf-8")

    n_train = n_val = 0
    for fpath in files:
        samples = build_samples_from_file(
            fpath, args.max_prefix_lines, args.max_suffix_lines,
            args.include_header, args.max_samples_per_file, rng
        )
        if not samples:
            continue

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
                if out is val_f: n_val += 1
                else: n_train += 1

    train_f.close(); val_f.close()
    print(f"Done. train={n_train}, val={n_val}")

if __name__ == "__main__":
    main()