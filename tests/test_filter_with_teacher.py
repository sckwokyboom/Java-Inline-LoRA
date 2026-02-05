import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import filter_with_teacher  # noqa: E402


class FilterWithTeacherTests(unittest.TestCase):
    def test_match_modes(self):
        ok, teacher_norm, gt_norm, sim = filter_with_teacher.compare_texts(
            gt_raw="foo();\n",
            teacher_raw="foo();\r\n",
            match_mode="exact",
            fuzzy_threshold=0.98,
            normalize_java_line_endings=True,
        )
        self.assertTrue(ok)
        self.assertEqual(teacher_norm, "foo();")
        self.assertEqual(gt_norm, "foo();")
        self.assertIsNone(sim)

        ok, _, _, _ = filter_with_teacher.compare_texts(
            gt_raw="foo();",
            teacher_raw="  foo();  ",
            match_mode="trimmed",
            fuzzy_threshold=0.98,
            normalize_java_line_endings=True,
        )
        self.assertTrue(ok)

        ok, _, _, _ = filter_with_teacher.compare_texts(
            gt_raw="foo(  a,   b );",
            teacher_raw="foo( a, b );",
            match_mode="ws_norm",
            fuzzy_threshold=0.98,
            normalize_java_line_endings=True,
        )
        self.assertTrue(ok)

        ok, _, _, sim = filter_with_teacher.compare_texts(
            gt_raw="return foo + bar;",
            teacher_raw="return foo+bar;",
            match_mode="fuzzy",
            fuzzy_threshold=0.90,
            normalize_java_line_endings=True,
        )
        self.assertTrue(ok)
        self.assertIsNotNone(sim)
        assert sim is not None
        self.assertGreaterEqual(sim, 0.90)

    def test_newline_is_truncated_to_single_line(self):
        ok, teacher_norm, gt_norm, _ = filter_with_teacher.compare_texts(
            gt_raw="foo();\n",
            teacher_raw="foo();\nbar();",
            match_mode="exact",
            fuzzy_threshold=0.98,
            normalize_java_line_endings=True,
        )
        self.assertTrue(ok)
        self.assertEqual(teacher_norm, "foo();")
        self.assertEqual(gt_norm, "foo();")

    def test_smoke_run_filter_with_mocked_client(self):
        samples = [
            {
                "id": "A::1",
                "file": "A.java",
                "line_index": 1,
                "prompt": "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
                "completion": "foo();\n",
            },
            {
                "id": "B::2",
                "file": "B.java",
                "line_index": 2,
                "prompt": "<|fim_prefix|>c<|fim_suffix|>d<|fim_middle|>",
                "completion": "bar();",
            },
            {
                "id": "C::3",
                "file": "C.java",
                "line_index": 3,
                "prompt": "<|fim_prefix|>e<|fim_suffix|>f<|fim_middle|>",
                "completion": "baz();",
            },
        ]
        responses = {
            samples[0]["prompt"]: "foo();\nbar();",
            samples[1]["prompt"]: "  bar();  ",
            samples[2]["prompt"]: "different();",
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            in_path = root / "in.jsonl"
            out_path = root / "out.jsonl"
            rejected_path = root / "rejected.jsonl"
            with in_path.open("w", encoding="utf-8") as f:
                for row in samples:
                    f.write(json.dumps(row))
                    f.write("\n")

            def fake_complete(self, prompt):
                return filter_with_teacher.CompletionResult(
                    text=responses[prompt],
                    latency_ms=12,
                    error=None,
                    error_reason=None,
                )

            args = SimpleNamespace(
                input_path=str(in_path),
                out=str(out_path),
                endpoint="http://localhost:9999/v1",
                model="teacher-model",
                rejected_out=str(rejected_path),
                max_keep=1500,
                max_samples=None,
                seed=42,
                shuffle=False,
                request_timeout=10.0,
                concurrency=1,
                qps=None,
                max_retries=0,
                retry_backoff_base=0.01,
                temperature=0.0,
                top_p=1.0,
                max_tokens=64,
                stop=["\n"],
                presence_penalty=None,
                frequency_penalty=None,
                match_mode="trimmed",
                fuzzy_threshold=0.98,
                normalize_java_line_endings=True,
                progress_every=1000,
            )

            with mock.patch.object(filter_with_teacher.TeacherClient, "complete", new=fake_complete):
                stats = filter_with_teacher.run_filter(args)

            kept_rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            rejected_rows = [
                json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]

            self.assertEqual(stats["total_processed"], 3)
            self.assertEqual(stats["total_kept"], 2)
            self.assertEqual(stats["total_rejected"], 1)
            self.assertEqual(len(kept_rows), 2)
            self.assertEqual(len(rejected_rows), 1)
            self.assertEqual(kept_rows[0]["teacher_filter"]["teacher_text_norm"], "foo();")
            self.assertEqual(rejected_rows[0]["teacher_filter"]["passed"], False)


if __name__ == "__main__":
    unittest.main()
