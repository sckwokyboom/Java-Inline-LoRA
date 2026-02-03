import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterator, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_len_lines(text: Optional[str]) -> Tuple[int, int]:
    if not text:
        return 0, 0
    return len(text), text.count("\n") + 1


def canonical_project_id(task_project: str, path_project: Optional[str]) -> str:
    task_project = (task_project or "").strip()
    if not task_project:
        return path_project or "unknown"
    if path_project:
        if task_project.replace("_", "/") == path_project:
            return path_project
        if task_project.replace("__", "/").replace("_", "/") == path_project:
            return path_project
    return task_project


def extract_project(sample: dict) -> Tuple[str, str, Optional[str]]:
    metadata = sample.get("metadata") or {}
    task_id = str(metadata.get("task_id") or "")
    task_project = task_id.split("/", 1)[0] if "/" in task_id else task_id

    fpath_tuple = metadata.get("fpath_tuple")
    path_project = None
    if isinstance(fpath_tuple, list) and fpath_tuple:
        first = str(fpath_tuple[0])
        second = str(fpath_tuple[1]) if len(fpath_tuple) > 1 else ""
        if second and ("/" in second or "-" in second or "_" in second):
            path_project = f"{first}/{second}"
        else:
            path_project = first

    return canonical_project_id(task_project, path_project), task_project or "unknown", path_project


def quantile(sorted_values: Sequence[int], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    n = len(sorted_values)
    idx = (n - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    w = idx - lo
    return sorted_values[lo] * (1 - w) + sorted_values[hi] * w


def summarize_distribution(values: Sequence[int]) -> dict:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    total = sum(sorted_values)
    count = len(sorted_values)
    mean = total / count
    return {
        "count": count,
        "min": sorted_values[0],
        "p25": quantile(sorted_values, 0.25),
        "p50": quantile(sorted_values, 0.50),
        "p75": quantile(sorted_values, 0.75),
        "p90": quantile(sorted_values, 0.90),
        "p95": quantile(sorted_values, 0.95),
        "p99": quantile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": mean,
    }


def format_summary(name: str, summary: dict) -> str:
    if summary.get("count", 0) == 0:
        return f"{name}: empty"
    keys = ["count", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"]
    parts = []
    for k in keys:
        v = summary.get(k)
        if isinstance(v, float) and not math.isfinite(v):
            continue
        if isinstance(v, float) and k != "mean":
            parts.append(f"{k}={v:.1f}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return f"{name}: " + ", ".join(parts)


def find_jsonl_files(paths: Sequence[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".jsonl":
            files.append(p)
            continue
        if p.is_dir():
            files.extend(sorted(p.rglob(pattern)))
    files = [f for f in files if f.is_file() and f.suffix == ".jsonl"]
    return sorted(set(files))


@dataclass
class SampleRow:
    file_name: str
    project: str
    task_project: str
    path_project: Optional[str]
    task_id: str
    prompt_chars: int
    prompt_lines: int
    gt_chars: int
    gt_lines: int


class RepoEvalStats:
    def __init__(self) -> None:
        self.total_samples = 0
        self.samples_by_project: Counter[str] = Counter()
        self.samples_by_task_project: Counter[str] = Counter()
        self.samples_by_file: Counter[str] = Counter()

        self.prompt_chars: list[int] = []
        self.prompt_lines: list[int] = []
        self.gt_chars: list[int] = []
        self.gt_lines: list[int] = []

        self.prompt_chars_by_project: DefaultDict[str, list[int]] = defaultdict(list)
        self.prompt_lines_by_project: DefaultDict[str, list[int]] = defaultdict(list)

        self.longest_prompts: list[SampleRow] = []

    def ingest_sample(self, file_name: str, sample: dict, keep_top_k: int) -> None:
        metadata = sample.get("metadata") or {}
        task_id = str(metadata.get("task_id") or "")

        project, task_project, path_project = extract_project(sample)

        prompt = sample.get("prompt")
        ground_truth = metadata.get("ground_truth")
        prompt_c, prompt_l = safe_len_lines(prompt)
        gt_c, gt_l = safe_len_lines(ground_truth)

        self.total_samples += 1
        self.samples_by_project[project] += 1
        self.samples_by_task_project[task_project] += 1
        self.samples_by_file[file_name] += 1

        self.prompt_chars.append(prompt_c)
        self.prompt_lines.append(prompt_l)
        self.gt_chars.append(gt_c)
        self.gt_lines.append(gt_l)

        self.prompt_chars_by_project[project].append(prompt_c)
        self.prompt_lines_by_project[project].append(prompt_l)

        row = SampleRow(
            file_name=file_name,
            project=project,
            task_project=task_project,
            path_project=path_project,
            task_id=task_id,
            prompt_chars=prompt_c,
            prompt_lines=prompt_l,
            gt_chars=gt_c,
            gt_lines=gt_l,
        )
        self._push_longest(row, keep_top_k)

    def _push_longest(self, row: SampleRow, keep_top_k: int) -> None:
        self.longest_prompts.append(row)
        self.longest_prompts.sort(key=lambda r: r.prompt_chars, reverse=True)
        if len(self.longest_prompts) > keep_top_k:
            self.longest_prompts = self.longest_prompts[:keep_top_k]

    def report_text(self, top_n_projects: int, top_k_longest: int) -> str:
        lines: list[str] = []
        lines.append(f"Total samples: {self.total_samples}")
        lines.append("")
        lines.append("By file:")
        for name, cnt in self.samples_by_file.most_common():
            share = (cnt / self.total_samples * 100) if self.total_samples else 0
            lines.append(f"  {name}: {cnt} ({share:.2f}%)")

        lines.append("")
        lines.append("By project (canonical):")
        for name, cnt in self.samples_by_project.most_common(top_n_projects):
            share = (cnt / self.total_samples * 100) if self.total_samples else 0
            lines.append(f"  {name}: {cnt} ({share:.2f}%)")

        lines.append("")
        if len(self.samples_by_task_project) != len(self.samples_by_project):
            lines.append("By project (task_id prefix):")
            for name, cnt in self.samples_by_task_project.most_common(top_n_projects):
                share = (cnt / self.total_samples * 100) if self.total_samples else 0
                lines.append(f"  {name}: {cnt} ({share:.2f}%)")
            lines.append("")

        lines.append(format_summary("Prompt chars", summarize_distribution(self.prompt_chars)))
        lines.append(format_summary("Prompt lines", summarize_distribution(self.prompt_lines)))
        lines.append(format_summary("Ground-truth chars", summarize_distribution(self.gt_chars)))
        lines.append(format_summary("Ground-truth lines", summarize_distribution(self.gt_lines)))

        lines.append("")
        lines.append("Per-project prompt chars (p50 / p90 / p99, count):")
        for name, _ in self.samples_by_project.most_common(top_n_projects):
            s = summarize_distribution(self.prompt_chars_by_project.get(name, []))
            if s.get("count", 0) == 0:
                continue
            lines.append(
                f"  {name}: p50={s['p50']:.1f}, p90={s['p90']:.1f}, p99={s['p99']:.1f}, count={s['count']}"
            )

        lines.append("")
        lines.append(f"Top-{top_k_longest} longest prompts (chars):")
        for i, row in enumerate(self.longest_prompts[:top_k_longest], 1):
            path_hint = row.path_project or ""
            lines.append(
                f"  {i}. chars={row.prompt_chars}, lines={row.prompt_lines}, project={row.project}, file={row.file_name}, task_id={row.task_id}, path={path_hint}"
            )
        return "\n".join(lines)

    def plot(self, output_dir: Path, max_projects: int) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        self._plot_hist(self.prompt_chars, "Prompt chars histogram", output_dir / "prompt_chars_hist.png")
        self._plot_hist(self.prompt_lines, "Prompt lines histogram", output_dir / "prompt_lines_hist.png")
        self._plot_bar_projects(self.samples_by_project, "Samples per project", output_dir / "samples_per_project.png",
                                max_projects)

    def _plot_hist(self, values: Sequence[int], title: str, out_path: Path) -> None:
        if not values:
            return
        plt.figure()
        plt.hist(values, bins=60)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    def _plot_bar_projects(self, counter: Counter[str], title: str, out_path: Path, max_projects: int) -> None:
        if not counter:
            return
        items = counter.most_common(max_projects)
        labels = [k for k, _ in items]
        counts = [v for _, v in items]
        plt.figure(figsize=(max(8, 0.45 * len(labels)), 5))
        plt.bar(range(len(labels)), counts)
        plt.title(title)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Samples")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Paths to directories or .jsonl files")
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--top-projects", type=int, default=50)
    parser.add_argument("--top-longest", type=int, default=20)
    parser.add_argument("--keep-top-k", type=int, default=200)
    parser.add_argument("--plots-dir", type=str, default="")
    parser.add_argument("--plots-max-projects", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p).expanduser().resolve() for p in args.paths]
    files = find_jsonl_files(input_paths, args.pattern)

    stats = RepoEvalStats()
    for file_path in files:
        file_name = file_path.name
        for sample in iter_jsonl(file_path):
            stats.ingest_sample(file_name, sample, keep_top_k=args.keep_top_k)

    print(stats.report_text(top_n_projects=args.top_projects, top_k_longest=args.top_longest))

    if args.plots_dir:
        stats.plot(Path(args.plots_dir).expanduser().resolve(), max_projects=args.plots_max_projects)


if __name__ == "__main__":
    main()
