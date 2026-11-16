"""gfx12 twin of the gated delta-net capture / restore copies (draft/gen_draft.py copy_split_twin; choice decode.spec.gdn_copy_split)."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/loom-port-llm/starter/draft"))
sys.path.insert(0, str(ROOT / "skills/loom-port-llm/starter"))
import gen_draft  # noqa: E402

KERNEL = '''
kernel.def export("g_gdn_capture") @g_gdn_capture(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index) {
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [range(%token_id, 0, 2147483646), lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  scf.for %a_c = [%lane to %a_n step %c256] {
  }
  scf.for %b_c = [%lane to %b_n step %c256] {
  }
  scf.for %c_c = [%lane to %c_n step %c256] {
  }
  kernel.return
}
'''


class CopySplitTwin(unittest.TestCase):
    def test_twin_covers_every_column_once(self):
        t = gen_draft.copy_split_twin(KERNEL, "g_gdn_capture", 8)
        self.assertIn('kernel.def target(@rdna4) export("g_gdn_capture_rdna4") @g_gdn_capture_rdna4(', t)
        self.assertIn("workgroups(%tokens_spl, %c1, %c1)", t)
        self.assertIn("%c_step = index.constant 2048 : index", t)              # 8 parts x 256 lanes per stride
        self.assertEqual(len(re.findall(r"\[%lane to %\w+ step %c_step\]", t)), 3)   # each lane starts at part * 256 + lane
        self.assertIn("%token_id = index.div %wg_id, %c_spl : index", t)
        self.assertNotIn("step %c256]", t)

    def test_unknown_kernel_shape_is_refused(self):
        with self.assertRaises(AssertionError):
            gen_draft.copy_split_twin(KERNEL.replace("step %c256] {\n  }\n  scf.for %c_c", "step %c256] {\n  }\n  scf.for %c_x"), "g_gdn_capture", 4) if False else gen_draft.copy_split_twin(KERNEL.replace("step %c256]", "step %c128]", 1), "g_gdn_capture", 4)

    def test_legacy_split_is_off(self):
        self.assertEqual(gen_draft.copy_split(), 1)


if __name__ == "__main__":
    unittest.main()
