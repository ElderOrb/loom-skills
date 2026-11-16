"""PROPOSE_HEAD_ROWS: the proposals' reduced-vocabulary Q4_0 head copy (draft/prep_weights.py subset converter, the policy, the
registry choice) and the row -> token id mapping a source-level model of g_head_select_p / the converter must agree with."""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTER = os.path.join(ROOT, "skills", "loom-port-llm", "starter")
sys.path.insert(0, os.path.join(STARTER, "draft"))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-kb"))
import prep_weights as pw  # noqa: E402
import choices  # noqa: E402

V, B, H = 248320, 248044, 5120


class Rows(unittest.TestCase):
    def test_subset_rows(self):
        self.assertEqual(pw.subset_rows(32768, B, V), (32768, B, 32768 + V - B))
        self.assertEqual(pw.subset_rows(V, B, V), (V, V, V))          # N = every row: a plain Q4_0 copy
        self.assertEqual(pw.subset_rows(B + 10, B, V), (B + 10, B + 10, V))   # N inside the added tokens: all rows from N on
        self.assertEqual(pw.subset_rows(10 ** 9, B, V), (V, V, V))

    def test_copy_map_is_the_row_subset(self):
        """Output block g reads source block g below split, g + skip from there on: exactly rows [0, N) + [B, V)."""
        for n in (32768, 65536, 1024):
            src, dst, blocks, (nbytes, split, skip) = pw.subset_entry("q6_k", "q4_0", n, B, V, H)
            per_row = H // 32
            self.assertEqual((src, dst), ("q6_k", "q4_0"))
            self.assertEqual(nbytes, V * H // 256 * 210)
            self.assertEqual(blocks, (n + V - B) * per_row)
            rows = [(g // per_row if g < split else (g + skip) // per_row) for g in range(0, blocks, per_row)]
            self.assertEqual(rows, list(range(n)) + list(range(B, V)))
            # within a row the 32-value group index is preserved
            for g in (0, split - 1, split, blocks - 1):
                s = g if g < split else g + skip
                self.assertEqual(s % per_row, g % per_row)

    def test_same_format_copy_counts_superblocks(self):
        """PROPOSE_HEAD=q6_k keeps the rows as they are: units are 256-value superblocks (210 bytes), same row map."""
        src, dst, blocks, (nbytes, split, skip) = pw.subset_entry("q6_k", "q6_k", 65536, B, V, H)
        per_row = H // 256
        self.assertEqual(blocks, (65536 + V - B) * per_row)
        rows = [(g // per_row if g < split else (g + skip) // per_row) for g in range(0, blocks, per_row)]
        self.assertEqual(rows, list(range(65536)) + list(range(B, V)))
        self.assertEqual(pw.dst_bytes("q6_k", blocks), blocks * 210)
        self.assertIn("view<210xi8>", pw.library({("q6_k", "q6_k")}))
        self.assertNotIn("g_q6_k_values", pw.library({("q6_k", "q6_k")}))

    def test_select_mapping(self):
        n, tail, rows = pw.subset_rows(32768, B, V)
        split, shift = n, tail - n
        to_id = lambda r: r if r < split else r + shift
        self.assertEqual([to_id(r) for r in (0, split - 1, split, rows - 1)], [0, 32767, B, V - 1])


class Policy(unittest.TestCase):
    def test_plain_policy_is_unchanged(self):
        """A drafter / Q8_0-head policy carries no row-subset constants."""
        t = pw.policy({"output.weight": ("q8_0", "q4_0", 1000)}, "x")
        self.assertNotIn("split_key", t)
        self.assertNotIn("%len_", t)

    def test_subset_policy(self):
        e = {"output.weight": pw.subset_entry("q6_k", "q4_0", 32768, B, V, H)}
        t = pw.policy(e, "x")
        for k in ("split_key", "skip_key", "%len_1", "%split_1", "%skip_1"):
            self.assertIn(k, t)
        self.assertIn(f"scalar.constant {e['output.weight'][3][0]} : i64", t)     # staged source bytes (the whole Q6_K tensor)
        self.assertIn(f"scalar.constant {e['output.weight'][2]} : i64", t)         # output blocks
        self.assertEqual(t.count("{"), t.count("}"))
        lib = pw.library({("q6_k", "q4_0")})
        self.assertIn("g_prep_q6_k_q4_0", lib)
        self.assertIn("llm.prep_split", lib)
        self.assertIn("g_q6_k_values", lib)
        self.assertEqual(lib.count("{"), lib.count("}"))
        self.assertNotIn("llm.prep_split", pw.library({("q8_0", "q4_0")}))

    def test_registry(self):
        c = choices.CHOICES["draft.propose.head_rows"]
        self.assertEqual((c.legacy, c.env, c.type), (0, "PROPOSE_HEAD_ROWS", "int"))


if __name__ == "__main__":
    unittest.main()
