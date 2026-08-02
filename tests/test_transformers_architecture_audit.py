from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TransformersArchitectureAuditTests(unittest.TestCase):
    def test_audit_matrix_is_separate_and_deterministic(self) -> None:
        fixture = json.loads(
            (ROOT / "tests" / "fixtures" / "transformers_tiny.json").read_text(
                encoding="utf-8"
            )
        )
        page = (
            ROOT / "tests" / "transformers-architecture-audit.html"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            [row["name"] for row in fixture["models"]],
            ["qwen2", "llama", "mistral", "gpt2", "bert", "phi3", "opt"],
        )
        deferred_rows = fixture["architecture_audit_models"]
        self.assertEqual(
            [row["name"] for row in deferred_rows],
            ["gemma2", "bloom", "t5"],
        )
        rows_by_name = {
            row["name"]: row
            for row in fixture["models"] + deferred_rows
        }
        audit_rows = [
            rows_by_name[name]
            for name in ("gemma2", "phi3", "opt", "bloom", "t5")
        ]
        self.assertEqual(
            [row["model_class"] for row in audit_rows],
            [
                "Gemma2ForCausalLM",
                "Phi3ForCausalLM",
                "OPTForCausalLM",
                "BloomForCausalLM",
                "T5ForConditionalGeneration",
            ],
        )
        self.assertEqual(audit_rows[0]["kwargs"]["num_hidden_layers"], 2)
        self.assertEqual(audit_rows[0]["kwargs"]["sliding_window"], 2)
        self.assertEqual(audit_rows[-1]["kwargs"]["num_decoder_layers"], 1)
        self.assertEqual(
            [row["expected_cpu_operator_count"] for row in audit_rows],
            [251, 99, 56, 88, 285],
        )
        self.assertEqual(
            [row["expected_cpu_distinct_operators"] for row in audit_rows],
            [37, 27, 17, 34, 46],
        )
        for row in audit_rows:
            self.assertTrue(row["required_cpu_operators"])
        self.assertIn('fixture["architecture_audit_models"]', page)
        self.assertIn('fixture["models"] +', page)
        self.assertIn("first_counts == second_counts", page)
        self.assertIn('"expected_cpu_operator_count"', page)
        self.assertIn("rtol=0.0, atol=0.0", page)
        self.assertIn('os.environ["TRANSFORMERS_OFFLINE"] = "1"', page)
        self.assertIn('importlib.util.find_spec("tokenizers") is None', page)


if __name__ == "__main__":
    unittest.main()
