import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Basic stopword set combining Java keywords and common English fillers
JAVA_STOPWORDS = {
    "public", "private", "protected", "class", "interface", "enum", "return", "static", "final",
    "void", "int", "long", "double", "float", "boolean", "char", "byte", "short", "new", "if",
    "else", "switch", "case", "default", "break", "continue", "for", "while", "do", "try",
    "catch", "finally", "throws", "throw", "extends", "implements", "this", "super", "null",
    "true", "false", "package", "import", "var", "assert", "instanceof", "yield", "record",
}
EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "with", "without", "on", "in", "at", "by", "to",
    "of", "for", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "that",
    "this", "these", "those",
}
STOPWORDS = JAVA_STOPWORDS | EN_STOPWORDS

TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9_.]+")
CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


@dataclass
class Chunk:
    chunk_id: int
    file: str  # repo-relative path, posix
    start: int  # 0-based inclusive
    end: int  # 0-based inclusive
    text: str
    lines: List[str]


def _split_identifier(token: str) -> List[str]:
    token = CAMEL_RE.sub(r"\1 \2", token)
    parts: List[str] = []
    for part in token.split("_"):
        if part:
            parts.append(part)
    return parts


def tokenize(text: str, drop_stopwords: bool = True, identifiers_only: bool = False) -> List[str]:
    tokens: List[str] = []
    for raw in TOKEN_SPLIT_RE.split(text):
        if not raw:
            continue
        pieces = _split_identifier(raw)
        for piece in pieces:
            piece_lower = piece.lower()
            if drop_stopwords and piece_lower in STOPWORDS:
                continue
            if identifiers_only and not re.match(r"^[a-z_][a-z0-9_]*$", piece_lower):
                continue
            tokens.append(piece_lower)
    return tokens


def truncate_snippet(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 5
    return text[:head] + "\n...\n" + text[-tail:]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq: Counter = Counter()
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.docs: List[List[str]] = []

    def add_documents(self, docs: List[List[str]]):
        for tokens in docs:
            self.docs.append(tokens)
            self.doc_len.append(len(tokens))
            for tok in set(tokens):
                self.doc_freq[tok] += 1
        if self.doc_len:
            self.avg_doc_len = sum(self.doc_len) / len(self.doc_len)

    def search(self, query_tokens: List[str], top_k: int = 20) -> List[Tuple[int, float]]:
        if not query_tokens or not self.docs:
            return []
        scores = defaultdict(float)
        total_docs = len(self.docs)
        q_freq = Counter(query_tokens)
        for tok, qf in q_freq.items():
            df = self.doc_freq.get(tok, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for doc_id, doc_tokens in enumerate(self.docs):
                tf = doc_tokens.count(tok)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / (self.avg_doc_len or 1))
                scores[doc_id] += idf * tf * (self.k1 + 1) / denom * qf
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


def _iter_line_windows(lines: List[str], size: int, overlap: int) -> Iterable[Tuple[int, int]]:
    if size <= 0:
        size = 30
    if overlap < 0:
        overlap = 0
    step = max(1, size - overlap)
    start = 0
    while start < len(lines):
        end = min(len(lines), start + size)
        yield start, end
        if end == len(lines):
            break
        start += step


def chunk_file_lines(path: Path, repo: Path, chunk_lines: int, chunk_overlap: int, chunk_id_start: int) -> Tuple[List[Chunk], int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)
    chunks: List[Chunk] = []
    for start, end in _iter_line_windows(lines, chunk_lines, chunk_overlap):
        chunk_lines_slice = lines[start:end]
        chunk_text = "".join(chunk_lines_slice)
        chunks.append(Chunk(
            chunk_id=chunk_id_start + len(chunks),
            file=str(path.relative_to(repo).as_posix()),
            start=start,
            end=end - 1 if end else start,
            text=chunk_text,
            lines=chunk_lines_slice,
        ))
    return chunks, chunk_id_start + len(chunks)


def _maybe_load_treesitter_java():
    try:
        from tree_sitter import Parser
        from tree_sitter_languages import get_language
    except Exception:
        return None, None
    lang = get_language("java")
    parser = Parser()
    parser.set_language(lang)
    return parser, lang


def _collect_ts_chunks(src: str, parser, path: Path, repo: Path, max_lines: int, overlap: int, chunk_id_start: int) -> Tuple[List[Chunk], int]:
    # Traverse AST to collect method/class/field bodies. Fallback to line windows for oversized nodes.
    tree = parser.parse(src.encode("utf-8"))
    root = tree.root_node
    targets = {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "method_declaration",
        "constructor_declaration",
        "field_declaration",
    }
    chunks: List[Chunk] = []
    stack = [root]
    lines = src.splitlines(keepends=True)
    while stack:
        cur = stack.pop()
        stack.extend(cur.children)
        if cur.type not in targets:
            continue
        start_line = cur.start_point[0]
        end_line = cur.end_point[0]
        if end_line < start_line:
            continue
        span_lines = lines[start_line:end_line + 1]
        if len(span_lines) > max_lines:
            # Break large nodes into overlapping windows to keep chunk sizes bounded.
            for start, end in _iter_line_windows(span_lines, max_lines, overlap):
                chunk_lines_slice = span_lines[start:end]
                chunks.append(Chunk(
                    chunk_id=chunk_id_start + len(chunks),
                    file=str(path.relative_to(repo).as_posix()),
                    start=start_line + start,
                    end=start_line + end - 1,
                    text="".join(chunk_lines_slice),
                    lines=chunk_lines_slice,
                ))
        else:
            chunks.append(Chunk(
                chunk_id=chunk_id_start + len(chunks),
                file=str(path.relative_to(repo).as_posix()),
                start=start_line,
                end=end_line,
                text="".join(span_lines),
                lines=span_lines,
            ))
    return chunks, chunk_id_start + len(chunks)


def build_chunks(files: Sequence[Path], repo: Path, chunker: str, chunk_lines: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 0
    use_ts = chunker == "treesitter"
    parser = None
    if use_ts:
        parser, _ = _maybe_load_treesitter_java()
        if parser is None:
            print("[rag] tree-sitter-java not available, falling back to line chunker")
            use_ts = False
    for path in files:
        if use_ts:
            src = path.read_text(encoding="utf-8", errors="ignore")
            ts_chunks, chunk_id = _collect_ts_chunks(src, parser, path, repo, chunk_lines, chunk_overlap, chunk_id)
            if ts_chunks:
                chunks.extend(ts_chunks)
                continue
        line_chunks, chunk_id = chunk_file_lines(path, repo, chunk_lines, chunk_overlap, chunk_id)
        chunks.extend(line_chunks)
    return chunks


def _line_set(start: int, end: int) -> Set[int]:
    return set(range(start, end + 1))


def _lines_match_completion(lines: Sequence[str], completion_line: str) -> bool:
    comp = completion_line.rstrip("\r\n")
    for ln in lines:
        if ln.rstrip("\r\n") == comp:
            return True
    return False


class RagRetriever:
    def __init__(
        self,
        repo: Path,
        files: Sequence[Path],
        *,
        k: int,
        max_chars: int,
        max_snippet_chars: int,
        method: str = "bm25",
        chunker: str = "lines",
        chunk_lines: int = 30,
        chunk_overlap: int = 10,
        query_mode: str = "hybrid",
        query_window_lines: int = 20,
        drop_stopwords: bool = True,
        use_identifiers_only: bool = False,
        exclude_same_file_window: int = 80,
        exclude_completion_text: bool = True,
        exclude_bench_targets: bool = True,
        excluded_by_file: Optional[Dict[str, Set[int]]] = None,
        text_blocklist_by_file: Optional[Dict[str, Set[str]]] = None,
        global_text_blocklist: Optional[Set[str]] = None,
    ):
        self.repo = repo
        self.k = k
        self.max_chars = max_chars
        self.max_snippet_chars = max_snippet_chars
        self.method = method
        self.chunker = chunker
        self.chunk_lines = chunk_lines
        self.chunk_overlap = chunk_overlap
        self.query_mode = query_mode
        self.query_window_lines = query_window_lines
        self.drop_stopwords = drop_stopwords
        self.use_identifiers_only = use_identifiers_only
        self.exclude_same_file_window = exclude_same_file_window
        self.exclude_completion_text = exclude_completion_text
        self.exclude_bench_targets = exclude_bench_targets
        self.excluded_by_file = excluded_by_file or {}
        self.text_blocklist_by_file = text_blocklist_by_file or {}
        self.global_text_blocklist = global_text_blocklist or set()
        self.exclusion_stats = defaultdict(int)

        if method != "bm25":
            raise ValueError(f"Unsupported rag method: {method}")
        self.chunks = build_chunks(files, repo, chunker, chunk_lines, chunk_overlap)
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        self.index = BM25Index()
        tokenized_docs = [tokenize(chunk.text, drop_stopwords=self.drop_stopwords, identifiers_only=False) for chunk in self.chunks]
        self.index.add_documents(tokenized_docs)
        if not self.chunks:
            print("[rag] WARNING: no chunks built for retrieval")

    def _masked_window_query(self, before: Sequence[str], after: Sequence[str]) -> str:
        before_tail = before[-self.query_window_lines:] if self.query_window_lines else before
        after_head = after[:self.query_window_lines] if self.query_window_lines else after
        return "".join(before_tail) + "\n" + "".join(after_head)

    def _identifier_query(self, before: Sequence[str], after: Sequence[str]) -> str:
        window = self._masked_window_query(before, after)
        identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", window)
        return " ".join(identifiers)

    def build_query(self, before: Sequence[str], after: Sequence[str]) -> str:
        mode = self.query_mode
        if mode == "masked_window":
            return self._masked_window_query(before, after)
        if mode == "ast_symbols":
            return self._identifier_query(before, after)
        if mode == "hybrid":
            return self._masked_window_query(before, after) + "\n" + self._identifier_query(before, after)
        return self._masked_window_query(before, after)

    def _passes_filters(self, chunk: Chunk, rel_path: str, line_index: int, completion_line: str) -> Optional[str]:
        if chunk.file == rel_path:
            if abs(chunk.start - line_index) <= self.exclude_same_file_window or abs(chunk.end - line_index) <= self.exclude_same_file_window:
                return "proximity"
            if chunk.start <= line_index <= chunk.end:
                return "proximity"
        if self.exclude_completion_text and _lines_match_completion(chunk.lines, completion_line):
            return "completion_text"
        if self.exclude_bench_targets:
            excluded_lines = self.excluded_by_file.get(chunk.file, set())
            if excluded_lines & _line_set(chunk.start, chunk.end):
                return "bench_line"
            blocklist_file = self.text_blocklist_by_file.get(chunk.file, set())
            for ln in chunk.lines:
                fp = ln.rstrip("\r\n")
                if fp in blocklist_file or fp in self.global_text_blocklist:
                    return "bench_text"
        return None

    def retrieve(self, rel_path: str, line_index: int, before: Sequence[str], after: Sequence[str], completion_line: str) -> Tuple[str, List[Dict], str]:
        query = self.build_query(before, after)
        query_tokens = tokenize(query, drop_stopwords=self.drop_stopwords, identifiers_only=self.use_identifiers_only)
        ranked = self.index.search(query_tokens, top_k=max(self.k * 6, 20))
        snippets: List[Tuple[Chunk, float]] = []
        seen_spans: Set[Tuple[str, int, int]] = set()
        for doc_id, score in ranked:
            chunk = self.chunk_by_id.get(doc_id)
            if chunk is None:
                continue
            span = (chunk.file, chunk.start, chunk.end)
            if span in seen_spans:
                self.exclusion_stats["dedupe"] += 1
                continue
            reason = self._passes_filters(chunk, rel_path, line_index, completion_line)
            if reason:
                self.exclusion_stats[reason] += 1
                continue
            snippets.append((chunk, score))
            seen_spans.add(span)
            if len(snippets) >= self.k * 3:
                break

        snippets.sort(key=lambda cs: cs[1], reverse=True)
        chosen = []
        total_chars = 0
        for chunk, score in snippets:
            snippet_text = truncate_snippet(chunk.text, self.max_snippet_chars)
            prospective = total_chars + len(snippet_text)
            if chosen and prospective > self.max_chars:
                break
            if not chosen and prospective > self.max_chars:
                snippet_text = truncate_snippet(chunk.text, self.max_chars)
            total_chars += len(snippet_text)
            chosen.append((chunk, snippet_text))
            if len(chosen) >= self.k:
                break

        if not chosen:
            return "", [], query

        lines = ["/* RAG_CONTEXT"]
        meta: List[Dict] = []
        for idx, (chunk, snippet_text) in enumerate(chosen, 1):
            lines.append(f"[{idx}] path: {chunk.file} (lines {chunk.start + 1}-{chunk.end + 1})")
            lines.append(snippet_text.rstrip("\n"))
            lines.append("")
            meta.append({"file": chunk.file, "start": chunk.start, "end": chunk.end})
        lines.append("END_RAG_CONTEXT */")
        rag_block = "\n".join(lines) + "\n"
        return rag_block, meta, query

    def report(self) -> Dict[str, int]:
        return dict(self.exclusion_stats)
