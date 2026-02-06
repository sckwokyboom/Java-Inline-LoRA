import argparse
import concurrent.futures
import gzip
import json
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional


RETRY_HTTP_CODES = {429, 502, 503, 504}
MATCH_MODES = {"exact", "trimmed", "ws_norm", "fuzzy"}


@dataclass
class CompletionResult:
    text: str
    latency_ms: Optional[int]
    error: Optional[str]
    error_reason: Optional[str]


@dataclass
class EvaluateResult:
    enriched_sample: dict
    passed: bool
    is_error: bool
    reason: str
    latency_ms: Optional[int]


class QpsLimiter:
    def __init__(self, qps: float) -> None:
        self.interval = 1.0 / qps
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_allowed:
                delay = self.next_allowed - now
                time.sleep(delay)
                now = time.monotonic()
            self.next_allowed = max(now, self.next_allowed) + self.interval


class TeacherClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        request_timeout: float,
        max_retries: int,
        retry_backoff_base: float,
        generation_params: dict[str, Any],
        qps_limiter: Optional[QpsLimiter] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.generation_params = generation_params
        self.qps_limiter = qps_limiter
        self.url = build_completions_url(self.endpoint)

    def complete(self, prompt: str) -> CompletionResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            **self.generation_params,
        }

        for attempt in range(self.max_retries + 1):
            if self.qps_limiter is not None:
                self.qps_limiter.acquire()

            t0 = time.perf_counter()
            try:
                text = self._request_once(payload)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return CompletionResult(
                    text=text,
                    latency_ms=latency_ms,
                    error=None,
                    error_reason=None,
                )
            except Exception as exc:  # noqa: BLE001
                reason, transient, detail = classify_error(exc)
                if (not transient) or attempt >= self.max_retries:
                    return CompletionResult(
                        text="",
                        latency_ms=None,
                        error=f"{reason}: {detail}",
                        error_reason=reason,
                    )

                delay = self.retry_backoff_base * (2 ** attempt) + random.uniform(0.0, self.retry_backoff_base)
                time.sleep(delay)

        return CompletionResult(
            text="",
            latency_ms=None,
            error="unknown_error: exhausted retries",
            error_reason="unknown_error",
        )

    def _request_once(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
        return parse_completion_response(data)


def build_completions_url(endpoint: str) -> str:
    ep = endpoint.rstrip("/")
    if ep.endswith("/completions"):
        return ep
    return f"{ep}/completions"


def parse_completion_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response_missing_choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("invalid_choice_type")

    text = first.get("text")
    if isinstance(text, str):
        return text

    # Minimal compatibility layer for endpoints that expose chat-like choice payloads.
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text_part = part.get("text")
                    if isinstance(text_part, str):
                        chunks.append(text_part)
            if chunks:
                return "".join(chunks)

    raise ValueError("response_missing_choices_text")


def classify_error(exc: Exception) -> tuple[str, bool, str]:
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
        try:
            payload = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            payload = ""
        detail = payload.strip()[:300] if payload else str(exc)
        return f"http_{code}", code in RETRY_HTTP_CODES, detail

    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, socket.timeout):
            return "timeout", True, str(reason)
        if isinstance(reason, ConnectionResetError):
            return "connection_reset", True, str(reason)
        reason_text = str(reason).lower()
        if "timed out" in reason_text:
            return "timeout", True, str(reason)
        return "network_error", True, str(reason)

    if isinstance(exc, socket.timeout):
        return "timeout", True, str(exc)

    if isinstance(exc, TimeoutError):
        return "timeout", True, str(exc)

    if isinstance(exc, ConnectionResetError):
        return "connection_reset", True, str(exc)

    if isinstance(exc, json.JSONDecodeError):
        return "bad_response_json", False, str(exc)

    if isinstance(exc, ValueError):
        return "bad_response_schema", False, str(exc)

    return "request_error", False, str(exc)


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode=mode, encoding="utf-8")
    return path.open(mode=mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open_text(path, "rt") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed JSON line {line_no} in {path}")
                continue
            if not isinstance(obj, dict):
                print(f"[warn] skipping non-object JSON line {line_no} in {path}")
                continue
            yield obj


def limited_samples(samples: Iterable[dict], max_samples: Optional[int]) -> Iterable[dict]:
    if max_samples is None or max_samples <= 0:
        yield from samples
        return
    for idx, sample in enumerate(samples):
        if idx >= max_samples:
            break
        yield sample


def reservoir_sample(samples: Iterable[dict], max_samples: int, seed: int) -> list[dict]:
    if max_samples <= 0:
        return []
    rnd = random.Random(seed)
    reservoir: list[dict] = []
    for idx, sample in enumerate(samples):
        if idx < max_samples:
            reservoir.append(sample)
            continue
        j = rnd.randint(0, idx)
        if j < max_samples:
            reservoir[j] = sample
    rnd.shuffle(reservoir)
    return reservoir


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n")


def take_first_line(text: str) -> str:
    if "\n" in text:
        return text.split("\n", 1)[0]
    if "\r" in text:
        return text.split("\r", 1)[0]
    return text


def ws_norm(text: str) -> str:
    return " ".join(text.split())


def normalize_for_mode(text: str, match_mode: str) -> str:
    if match_mode == "exact":
        return text.rstrip("\r\n")
    if match_mode == "trimmed":
        return text.strip()
    if match_mode in {"ws_norm", "fuzzy"}:
        return ws_norm(text)
    raise ValueError(f"unknown match_mode={match_mode}")


def compare_texts(
    gt_raw: str,
    teacher_raw: str,
    match_mode: str,
    fuzzy_threshold: float,
    normalize_java_line_endings: bool,
) -> tuple[bool, str, str, Optional[float]]:
    gt = gt_raw
    teacher = teacher_raw
    if normalize_java_line_endings:
        gt = normalize_line_endings(gt)
        teacher = normalize_line_endings(teacher)

    gt = take_first_line(gt)
    teacher = take_first_line(teacher)

    gt_norm = normalize_for_mode(gt, match_mode)
    teacher_norm = normalize_for_mode(teacher, match_mode)

    if match_mode == "fuzzy":
        similarity = SequenceMatcher(None, gt_norm, teacher_norm).ratio()
        return similarity >= fuzzy_threshold, teacher_norm, gt_norm, similarity
    return teacher_norm == gt_norm, teacher_norm, gt_norm, None


def normalized_gt_for_metadata(completion: str, match_mode: str, normalize_java_line_endings: bool) -> str:
    gt = completion
    if normalize_java_line_endings:
        gt = normalize_line_endings(gt)
    gt = take_first_line(gt)
    return normalize_for_mode(gt, match_mode)


def evaluate_sample(
    sample: dict,
    client: TeacherClient,
    endpoint: str,
    model: str,
    generation_params: dict[str, Any],
    match_mode: str,
    fuzzy_threshold: float,
    normalize_java_line_endings: bool,
) -> EvaluateResult:
    prompt = sample.get("prompt")
    completion = sample.get("completion")

    if not isinstance(prompt, str):
        gt_norm = (
            normalized_gt_for_metadata(completion, match_mode, normalize_java_line_endings)
            if isinstance(completion, str)
            else ""
        )
        meta = build_teacher_filter(
            endpoint=endpoint,
            model=model,
            generation_params=generation_params,
            match_mode=match_mode,
            passed=False,
            teacher_text="",
            teacher_text_norm="",
            gt_norm=gt_norm,
            similarity=None,
            latency_ms=None,
            error="bad_input: missing prompt string",
        )
        out = dict(sample)
        out["teacher_filter"] = meta
        return EvaluateResult(out, False, True, "error:bad_input", None)

    if not isinstance(completion, str):
        meta = build_teacher_filter(
            endpoint=endpoint,
            model=model,
            generation_params=generation_params,
            match_mode=match_mode,
            passed=False,
            teacher_text="",
            teacher_text_norm="",
            gt_norm="",
            similarity=None,
            latency_ms=None,
            error="bad_input: missing completion string",
        )
        out = dict(sample)
        out["teacher_filter"] = meta
        return EvaluateResult(out, False, True, "error:bad_input", None)

    result = client.complete(prompt)
    if result.error is not None:
        gt_norm = normalized_gt_for_metadata(completion, match_mode, normalize_java_line_endings)
        meta = build_teacher_filter(
            endpoint=endpoint,
            model=model,
            generation_params=generation_params,
            match_mode=match_mode,
            passed=False,
            teacher_text="",
            teacher_text_norm="",
            gt_norm=gt_norm,
            similarity=None,
            latency_ms=result.latency_ms,
            error=result.error,
        )
        out = dict(sample)
        out["teacher_filter"] = meta
        reason = f"error:{result.error_reason or 'request_error'}"
        return EvaluateResult(out, False, True, reason, result.latency_ms)

    passed, teacher_norm, gt_norm, similarity = compare_texts(
        gt_raw=completion,
        teacher_raw=result.text,
        match_mode=match_mode,
        fuzzy_threshold=fuzzy_threshold,
        normalize_java_line_endings=normalize_java_line_endings,
    )

    meta = build_teacher_filter(
        endpoint=endpoint,
        model=model,
        generation_params=generation_params,
        match_mode=match_mode,
        passed=passed,
        teacher_text=result.text,
        teacher_text_norm=teacher_norm,
        gt_norm=gt_norm,
        similarity=similarity,
        latency_ms=result.latency_ms,
        error=None,
    )
    out = dict(sample)
    out["teacher_filter"] = meta
    return EvaluateResult(out, passed, False, "passed" if passed else "mismatch", result.latency_ms)


def build_teacher_filter(
    endpoint: str,
    model: str,
    generation_params: dict[str, Any],
    match_mode: str,
    passed: bool,
    teacher_text: str,
    teacher_text_norm: str,
    gt_norm: str,
    similarity: Optional[float],
    latency_ms: Optional[int],
    error: Optional[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": model,
        "endpoint": endpoint,
        "params": generation_params,
        "match_mode": match_mode,
        "passed": passed,
        "teacher_text": teacher_text,
        "teacher_text_norm": teacher_text_norm,
        "gt_norm": gt_norm,
        "error": error,
    }
    if similarity is not None:
        out["similarity"] = similarity
    if latency_ms is not None:
        out["latency_ms"] = latency_ms
    return out


def dump_jsonl_line(fh, obj: dict) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False))
    fh.write("\n")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_path_for_output(out_path: Path) -> Path:
    name = out_path.name
    if name.endswith(".jsonl.gz"):
        stem = name[: -len(".jsonl.gz")]
    elif name.endswith(".jsonl"):
        stem = name[: -len(".jsonl")]
    else:
        stem = out_path.stem
    return out_path.with_name(f"{stem}.run_manifest.json")


def get_git_commit(cwd: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return None
    value = proc.stdout.strip()
    return value or None


def write_manifest(
    out_path: Path,
    args: argparse.Namespace,
    start_ts: str,
    end_ts: str,
    stats: dict[str, Any],
) -> Path:
    manifest = {
        "start_timestamp_utc": start_ts,
        "end_timestamp_utc": end_ts,
        "args": vars(args),
        "argv": sys.argv,
        "git_commit": get_git_commit(Path.cwd()),
        "python_version": sys.version,
        "model": args.model,
        "endpoint": args.endpoint,
        **stats,
    }
    path = manifest_path_for_output(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def load_samples(input_path: Path, shuffle: bool, seed: int, max_samples: Optional[int]) -> Iterable[dict]:
    raw_iter = iter_jsonl(input_path)
    if not shuffle:
        return limited_samples(raw_iter, max_samples)

    if max_samples is not None and max_samples > 0:
        # Use reservoir sampling to avoid loading the full dataset into memory.
        return reservoir_sample(raw_iter, max_samples, seed)

    print("[warn] --shuffle enabled without --max_samples; loading full input into memory")
    all_samples = list(raw_iter)
    rnd = random.Random(seed)
    rnd.shuffle(all_samples)
    return all_samples


def run_filter(args: argparse.Namespace) -> dict[str, Any]:
    if args.match_mode not in MATCH_MODES:
        raise ValueError(f"Unsupported match mode: {args.match_mode}")

    input_path = Path(args.input_path)
    out_path = Path(args.out)
    rejected_path = Path(args.rejected_out) if args.rejected_out else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rejected_path is not None:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)

    if args.max_keep is not None and args.max_keep <= 0:
        max_keep = None
    else:
        max_keep = args.max_keep

    if args.max_samples is not None and args.max_samples <= 0:
        max_samples = None
    else:
        max_samples = args.max_samples

    generation_params: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop": args.stop if args.stop else ["\n"],
    }
    if args.presence_penalty is not None:
        generation_params["presence_penalty"] = args.presence_penalty
    if args.frequency_penalty is not None:
        generation_params["frequency_penalty"] = args.frequency_penalty

    limiter = QpsLimiter(args.qps) if args.qps is not None and args.qps > 0 else None
    client = TeacherClient(
        endpoint=args.endpoint,
        model=args.model,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_base=args.retry_backoff_base,
        generation_params=generation_params,
        qps_limiter=limiter,
    )

    samples_iter = iter(load_samples(input_path, args.shuffle, args.seed, max_samples))

    processed = 0
    kept = 0
    rejected = 0
    errors = 0
    in_count = 0
    latency_sum = 0.0
    latency_count = 0
    reasons = Counter()

    start_ts = now_utc_iso()
    reached_max_keep = False
    reached_max_samples = False

    with open_text(out_path, "wt") as out_f:
        if rejected_path is not None:
            rejected_fh_ctx = open_text(rejected_path, "wt")
        else:
            rejected_fh_ctx = None

        try:
            with rejected_fh_ctx if rejected_fh_ctx is not None else _null_context() as rejected_f:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    inflight: dict[concurrent.futures.Future, dict] = {}
                    input_exhausted = False

                    def submit_more() -> None:
                        nonlocal input_exhausted, in_count
                        while not input_exhausted and len(inflight) < args.concurrency:
                            try:
                                sample = next(samples_iter)
                            except StopIteration:
                                input_exhausted = True
                                break
                            in_count += 1
                            fut = executor.submit(
                                evaluate_sample,
                                sample,
                                client,
                                args.endpoint,
                                args.model,
                                generation_params,
                                args.match_mode,
                                args.fuzzy_threshold,
                                args.normalize_java_line_endings,
                            )
                            inflight[fut] = sample

                    submit_more()

                    while inflight:
                        done, _ = concurrent.futures.wait(
                            inflight.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for fut in done:
                            submitted_sample = inflight.pop(fut, None)
                            try:
                                result: EvaluateResult = fut.result()
                            except Exception as exc:  # noqa: BLE001
                                errors += 1
                                processed += 1
                                rejected += 1
                                reasons["error:worker_exception"] += 1
                                if rejected_f is not None:
                                    base_obj = dict(submitted_sample) if isinstance(submitted_sample, dict) else {}
                                    base_obj["teacher_filter"] = {
                                        "model": args.model,
                                        "endpoint": args.endpoint,
                                        "params": generation_params,
                                        "match_mode": args.match_mode,
                                        "passed": False,
                                        "teacher_text": "",
                                        "teacher_text_norm": "",
                                        "gt_norm": "",
                                        "error": f"worker_exception: {exc}",
                                    }
                                    dump_jsonl_line(
                                        rejected_f,
                                        base_obj,
                                    )
                                continue

                            processed += 1
                            reasons[result.reason] += 1
                            if result.latency_ms is not None:
                                latency_sum += result.latency_ms
                                latency_count += 1

                            if result.passed:
                                kept += 1
                                dump_jsonl_line(out_f, result.enriched_sample)
                            else:
                                rejected += 1
                                if result.is_error:
                                    errors += 1
                                if rejected_f is not None:
                                    dump_jsonl_line(rejected_f, result.enriched_sample)

                            if max_keep is not None and kept >= max_keep:
                                reached_max_keep = True
                                break

                            if args.progress_every > 0 and processed % args.progress_every == 0:
                                keep_rate = (kept / processed) if processed else 0.0
                                avg_latency = (latency_sum / latency_count) if latency_count else 0.0
                                input_submitted_display = (
                                    f"{in_count}/{max_samples}" if max_samples is not None else f"{in_count}"
                                )
                                kept_display = f"{kept}/{max_keep}" if max_keep is not None else f"{kept}"
                                print(
                                    "[progress] "
                                    f"processed={processed} "
                                    f"input_submitted={input_submitted_display} "
                                    f"kept={kept_display} "
                                    f"rejected={rejected} "
                                    f"errors={errors} "
                                    f"keep_rate={keep_rate:.4f} "
                                    f"avg_latency_ms={avg_latency:.1f}"
                                )

                        if reached_max_keep:
                            for pending in list(inflight.keys()):
                                pending.cancel()
                            break

                        submit_more()
        finally:
            pass

    end_ts = now_utc_iso()

    keep_rate = (kept / processed) if processed else 0.0
    avg_latency = (latency_sum / latency_count) if latency_count else 0.0

    print("[summary]")
    print(f"  processed={processed}")
    input_submitted_display = f"{in_count}/{max_samples}" if max_samples is not None else f"{in_count}"
    kept_display = f"{kept}/{max_keep}" if max_keep is not None else f"{kept}"
    print(f"  input_submitted={input_submitted_display}")
    print(f"  kept={kept_display}")
    print(f"  rejected={rejected}")
    print(f"  errors={errors}")
    print(f"  keep_rate={keep_rate:.4f}")
    print(f"  avg_latency_ms={avg_latency:.1f}")
    if reached_max_keep:
        print(f"  stopped_early=max_keep({max_keep}) reached")
    if max_samples is not None and in_count >= max_samples:
        reached_max_samples = True
        print(f"  stopped_early=max_samples({max_samples}) reached")

    print("[failure_reasons]")
    for reason, cnt in reasons.most_common():
        if reason == "passed":
            continue
        print(f"  {reason}: {cnt}")

    stats = {
        "total_processed": processed,
        "total_kept": kept,
        "total_rejected": rejected,
        "total_errors": errors,
        "keep_rate": keep_rate,
        "avg_latency_ms": avg_latency,
        "failure_reasons": dict(reasons),
        "stopped_early_max_keep": reached_max_keep,
        "stopped_early_max_samples": reached_max_samples,
    }
    manifest_path = write_manifest(out_path, args, start_ts=start_ts, end_ts=end_ts, stats=stats)
    print(f"[manifest] wrote {manifest_path}")
    return stats


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter FIM Java samples using a teacher model served via /completions.")
    p.add_argument("--in", dest="input_path", required=True, help="Input JSONL(.gz) dataset path")
    p.add_argument("--out", required=True, help="Output JSONL(.gz) path for kept samples")
    p.add_argument("--endpoint", required=True, help="OpenAI-compatible base endpoint, e.g. http://host:port/v1")
    p.add_argument("--model", required=True, help="Model name used by the teacher endpoint")

    p.add_argument("--rejected_out", default=None, help="Optional JSONL(.gz) path for rejected/errored samples")
    p.add_argument(
        "--max_keep",
        type=int,
        default=None,
        help="Stop after this many kept samples. Omit for no limit. <=0 means no limit.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum input samples to process. Omit for no limit. <=0 means no limit.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle")
    p.add_argument("--shuffle", action="store_true", help="Shuffle input sample order before processing")
    p.add_argument("--request_timeout", type=float, default=60.0, help="Request timeout in seconds")
    p.add_argument("--concurrency", type=int, default=8, help="Number of parallel requests")
    p.add_argument("--qps", type=float, default=None, help="Optional global request starts per second")
    p.add_argument("--max_retries", type=int, default=5, help="Max retries on transient failures")
    p.add_argument("--retry_backoff_base", type=float, default=0.5, help="Exponential backoff base seconds")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--stop", action="append", default=None, help="Repeatable stop sequence; default is '\\n'")
    p.add_argument("--presence_penalty", type=float, default=None)
    p.add_argument("--frequency_penalty", type=float, default=None)

    p.add_argument("--match_mode", choices=sorted(MATCH_MODES), default="ws_norm")
    p.add_argument("--fuzzy_threshold", type=float, default=0.98)
    p.add_argument(
        "--normalize_java_line_endings",
        dest="normalize_java_line_endings",
        action="store_true",
        default=True,
        help="Normalize CRLF to LF before post-processing/matching (default: true)",
    )
    p.add_argument(
        "--no-normalize_java_line_endings",
        dest="normalize_java_line_endings",
        action="store_false",
        help="Disable CRLF->LF normalization",
    )

    p.add_argument("--progress_every", type=int, default=50, help="Print progress every N processed samples")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.concurrency <= 0:
        parser.error("--concurrency must be >= 1")
    if args.request_timeout <= 0:
        parser.error("--request_timeout must be > 0")
    if args.max_retries < 0:
        parser.error("--max_retries must be >= 0")
    if args.fuzzy_threshold < 0 or args.fuzzy_threshold > 1:
        parser.error("--fuzzy_threshold must be between 0 and 1")
    if args.qps is not None and args.qps <= 0:
        parser.error("--qps must be > 0")

    try:
        run_filter(args)
    except KeyboardInterrupt:
        print("[error] interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
