import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_codefix_dataset  # noqa: E402
import train_codefix_lora  # noqa: E402


class CodefixDatasetTests(unittest.TestCase):
    def test_extract_invocation_args_simple(self):
        args = make_codefix_dataset.extract_invocation_args("a, b")
        self.assertEqual(args, ["a", "b"])

        parsed = make_codefix_dataset.parse_invocation_statement("obj.call(a, b);\n")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(list(parsed.args), ["a", "b"])

    def test_extract_invocation_args_nested(self):
        parsed = make_codefix_dataset.parse_invocation_statement(
            "foo(bar(a,b), map.get(k));\n"
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(list(parsed.args), ["bar(a,b)", "map.get(k)"])

    def test_mutation_swap_args(self):
        parsed = make_codefix_dataset.parse_invocation_statement("obj.call(a, b, c);\n")
        self.assertIsNotNone(parsed)
        assert parsed is not None

        rng = make_codefix_dataset._candidate_rng(42, "A.java", 10, "swap_args")
        mutated = make_codefix_dataset.mutate_statement(parsed, "swap_args", rng)
        self.assertIsNotNone(mutated)
        assert mutated is not None

        self.assertNotEqual(mutated, parsed.line_text)
        self.assertTrue(mutated.startswith("obj.call("))
        self.assertTrue(mutated.endswith(");"))

        reparsed = make_codefix_dataset.parse_invocation_statement(mutated)
        self.assertIsNotNone(reparsed)
        assert reparsed is not None
        self.assertEqual(len(reparsed.args), len(parsed.args))
        self.assertCountEqual(list(reparsed.args), list(parsed.args))

    def test_mutation_missing_arg(self):
        parsed = make_codefix_dataset.parse_invocation_statement("obj.call(a, b, c);\n")
        self.assertIsNotNone(parsed)
        assert parsed is not None

        rng = make_codefix_dataset._candidate_rng(42, "A.java", 10, "missing_arg")
        mutated = make_codefix_dataset.mutate_statement(parsed, "missing_arg", rng)
        self.assertIsNotNone(mutated)
        assert mutated is not None

        reparsed = make_codefix_dataset.parse_invocation_statement(mutated)
        self.assertIsNotNone(reparsed)
        assert reparsed is not None
        self.assertEqual(len(reparsed.args), len(parsed.args) - 1)

    def test_mutation_extra_arg(self):
        parsed = make_codefix_dataset.parse_invocation_statement("obj.call(a, b);\n")
        self.assertIsNotNone(parsed)
        assert parsed is not None

        rng = make_codefix_dataset._candidate_rng(42, "A.java", 10, "extra_arg")
        mutated = make_codefix_dataset.mutate_statement(parsed, "extra_arg", rng)
        self.assertIsNotNone(mutated)
        assert mutated is not None

        reparsed = make_codefix_dataset.parse_invocation_statement(mutated)
        self.assertIsNotNone(reparsed)
        assert reparsed is not None
        self.assertEqual(len(reparsed.args), len(parsed.args) + 1)

    def test_bm25_excludes_self(self):
        statements = [
            make_codefix_dataset.StatementEntry(file="A.java", line_index=10, text="obj.call(a, b);"),
            make_codefix_dataset.StatementEntry(file="A.java", line_index=12, text="obj.call(x, y);"),
            make_codefix_dataset.StatementEntry(file="B.java", line_index=3, text="service.call(a, b, c);"),
            make_codefix_dataset.StatementEntry(file="C.java", line_index=8, text="obj.execute(task, ctx);"),
        ]
        retriever = make_codefix_dataset.StatementBM25Retriever(
            statements,
            same_file_window=5,
            drop_stopwords=True,
            use_identifiers_only=False,
        )

        candidate = make_codefix_dataset.CodefixCandidate(
            file="A.java",
            line_index=10,
            original_line="obj.call(a, b);",
            broken_line="obj.call(b, a);",
            mutation_type="swap_args",
            compiler_problems=("argument order mismatch in method invocation",),
        )
        augmentations = retriever.retrieve(candidate, top_k=3)

        self.assertTrue(augmentations)
        self.assertNotIn("obj.call(a, b);", augmentations)
        self.assertNotIn("obj.call(b, a);", augmentations)

    def test_prompt_and_completion_contract(self):
        candidate = make_codefix_dataset.CodefixCandidate(
            file="A.java",
            line_index=7,
            original_line="obj.call(a, b);",
            broken_line="obj.call(b, a);",
            mutation_type="swap_args",
            compiler_problems=("argument order mismatch in method invocation",),
        )
        row = make_codefix_dataset.build_record(
            candidate,
            augmentations=["service.call(userId, payload);", "validator.check(a, b);"]
        )

        self.assertIn("With these similar correct statements", row["prompt"])
        self.assertIn("Fix this problems", row["prompt"])
        self.assertIn("In this code snippet to fix", row["prompt"])
        self.assertTrue(row["completion"].startswith("```java\n"))
        self.assertTrue(row["completion"].endswith("\n```"))
        self.assertEqual(row["completion"], "```java\nobj.call(a, b);\n```")

    def test_exact_5000_split_by_file(self):
        pools = {m: [] for m in make_codefix_dataset.MUTATION_TYPES}
        for file_idx in range(50):
            file_name = f"F{file_idx}.java"
            for line_idx in range(120):
                for mutation_type in make_codefix_dataset.MUTATION_TYPES:
                    pools[mutation_type].append(
                        make_codefix_dataset.CodefixCandidate(
                            file=file_name,
                            line_index=line_idx,
                            original_line="obj.call(a, b);",
                            broken_line="obj.call(b, a);",
                            mutation_type=mutation_type,
                            compiler_problems=(make_codefix_dataset.PROBLEM_BY_MUTATION[mutation_type],),
                        )
                    )

        targets = make_codefix_dataset.target_counts_by_mutation(5000)
        train, val = make_codefix_dataset.choose_candidates_with_split(
            pools=pools,
            targets=targets,
            total_samples=5000,
            val_count=100,
            seed=42,
            max_attempts=200,
        )

        self.assertEqual(len(train), 4900)
        self.assertEqual(len(val), 100)

        train_files = {item.file for item in train}
        val_files = {item.file for item in val}
        self.assertFalse(train_files & val_files)

    def test_fail_when_not_enough_candidates(self):
        pools = {
            "swap_args": [
                make_codefix_dataset.CodefixCandidate(
                    file="A.java",
                    line_index=1,
                    original_line="obj.call(a, b);",
                    broken_line="obj.call(b, a);",
                    mutation_type="swap_args",
                    compiler_problems=(make_codefix_dataset.PROBLEM_BY_MUTATION["swap_args"],),
                )
            ],
            "missing_arg": [],
            "extra_arg": [],
        }
        targets = make_codefix_dataset.target_counts_by_mutation(6)
        with self.assertRaises(RuntimeError):
            make_codefix_dataset.choose_candidates_with_split(
                pools=pools,
                targets=targets,
                total_samples=6,
                val_count=1,
                seed=42,
                max_attempts=5,
            )

    def test_trainer_dry_run_smoke(self):
        class DummyTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.eos_token = "<eos>"
                self.pad_token = None

            def _encode(self, text):
                if not text:
                    return [2]
                return [2 + (ord(ch) % 13) for ch in text[:32]]

            def __call__(self, text, add_special_tokens=False):
                if isinstance(text, list):
                    return {"input_ids": [self._encode(t) for t in text]}
                return {"input_ids": self._encode(text)}

            def save_pretrained(self, out):
                return None

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=True)
                self.device = torch.device("cpu")

            def named_modules(self, memo=None, prefix="", remove_duplicate=True):
                modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
                for name in modules:
                    yield name, torch.nn.Identity()

            def gradient_checkpointing_enable(self):
                return None

            def print_trainable_parameters(self):
                return None

            def save_pretrained(self, out):
                return None

            def forward(self, input_ids=None, attention_mask=None, labels=None):
                return SimpleNamespace(loss=torch.tensor(0.0))

        fake_ds = {
            "train": [
                {"prompt": "<|im_start|>system\nS\n<|im_end|>\n<|im_start|>assistant\n", "completion": "```java\nfoo();\n```"}
            ],
            "validation": [
                {"prompt": "<|im_start|>system\nS\n<|im_end|>\n<|im_start|>assistant\n", "completion": "```java\nbar();\n```"}
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            train_path = td_path / "train.jsonl"
            val_path = td_path / "val.jsonl"
            train_path.write_text("{}\n", encoding="utf-8")
            val_path.write_text("{}\n", encoding="utf-8")

            argv = [
                "train_codefix_lora.py",
                "--train",
                str(train_path),
                "--val",
                str(val_path),
                "--out",
                str(td_path / "adapter"),
                "--dry_run",
                "--batch_size",
                "1",
                "--grad_accum",
                "1",
            ]

            with mock.patch.object(train_codefix_lora, "_prepare_datasets", return_value=(fake_ds, {"loaded": 2, "kept": 2, "dropped_missing": 0})), \
                 mock.patch.object(train_codefix_lora.AutoTokenizer, "from_pretrained", return_value=DummyTokenizer()), \
                 mock.patch.object(train_codefix_lora.AutoModelForCausalLM, "from_pretrained", return_value=DummyModel()), \
                 mock.patch.object(train_codefix_lora, "get_peft_model", side_effect=lambda model, cfg: model), \
                 mock.patch.object(sys, "argv", argv):
                train_codefix_lora.main()


if __name__ == "__main__":
    unittest.main()
