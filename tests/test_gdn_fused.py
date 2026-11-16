"""gfx12 fused GDN prefill (gen_gdn_fused.py): kernel text, gfx1201 compile, opt-in program wiring."""
import json, os, re, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "loom-port-llm" / "starter"))
sys.path.insert(0, str(ROOT / "skills" / "loom-port-llm" / "starter" / "draft"))
import gen_gdn as GG  # noqa: E402
import gen_gdn_fused as GF  # noqa: E402
import gen_draft  # noqa: E402

CFG = {"gdn": {"conv_kernel": 4, "k_heads": 16, "v_heads": 48, "head_k": 128, "head_v": 128, "inner": 6144, "v_order": "grouped"},
       "recurrent_layers": [i for i in range(64) if i % 4 != 3]}


def loom_compile():
    for c in (os.environ.get("LOOM_COMPILE"), shutil.which("loom-compile"), str(Path.home() / "loom/.venv/bin/loom-compile")):
        if c and os.path.exists(c):
            return c
    return None


class FusedGdn(unittest.TestCase):
    def kernels(self, spec=False):
        G = GG.gdn_geometry(CFG)
        go = "    %publish = func.call @g_gdn_publish(%metadata, %spans, %span) : (buffer, index, index) -> (i1)\n    scf.if %publish {\n" if spec else ""
        return G, GG.SLOT + GF.fused_kernels(G, 48, G["layer_bytes"], 1e-6, True, 3, go, "\n    }" if spec else "")

    def test_kernels_and_ablations(self):
        G, t = self.kernels()
        for k in ("g_gdn_fa_rdna4", "g_gdn_fb_rdna4"):
            self.assertEqual(t.count(f'export("{k}")'), 1)
        for a in (("nosolve",), ("nostore",), ("noconv",)):
            self.assertIn(f"g_gdn_fa_rdna4_{a[0]}", GF.phase_a(G, 48, G["layer_bytes"], 1e-6, True, 3, abl=a))
        self.assertIn("g_gdn_fb_rdna4_nonorm", GF.phase_b(G, 48, G["layer_bytes"], 1e-6, True, 3, abl=("nonorm",)))

    def test_fused_mode_is_opt_in(self):
        G = GG.gdn_geometry(CFG)
        old = os.environ.pop("GDN_FUSED2", None)
        try:
            self.assertFalse(GG.fused_mode(G))
            os.environ["GDN_FUSED2"] = "1"
            self.assertTrue(GG.fused_mode(G))
        finally:
            os.environ.pop("GDN_FUSED2", None)
            if old is not None:
                os.environ["GDN_FUSED2"] = old

    def test_verify_captures_both_branches(self):
        """A verify epoch can pack another request's prompt rows with the speculative spans (t >= 64): the fused GEMM branch's replay
        inputs are captured too, once per layer and branch (the marker), the chain branch keeps its capture before g_gdn_conv."""
        conv = "kernel.launch @g_gdn_conv[%t](%t, %spans, %layer_3, %metadata, %info, %origins, %arena, %gq, %conv_3, %gs) : [index](...)"
        body = (f"  scf.if %gemm {{\n    // gdn-capture: %layer_3\n    // dual-group: rdna4 kernel.launch @g_gdn_fa_rdna4[%t, %spans](...)\n"
                f"    {conv}\n    // dual-group: end\n  }} else {{\n  {conv}\n  }}")
        out = gen_draft.gdn_capture(body)
        self.assertEqual(out.count("@g_gdn_capture[%t](%t, %layer_3,"), 2)
        lines = out.split("\n")
        i = lines.index("    // gdn-capture: %layer_3")
        self.assertEqual(lines[i - 3:i], ["    scf.if %not_gated {", lines[i - 2], "    }"])
        self.assertTrue(lines[i - 2].startswith("      kernel.launch @g_gdn_capture["))
        self.assertEqual(lines[1], "    scf.if %not_gated {")   # before the dual group, not inside it
        self.assertEqual(gen_draft.gdn_capture("  " + conv).count("@g_gdn_capture["), 1)

    def test_workspace_slots_distinct_any_packing(self):
        """Slots floor(first / 32) + rank + chunk (rank by first row among live spans) never collide and stay below
        ceil(t / 32) + spans + 1, whatever order the packets list the spans in."""
        import random
        rnd = random.Random(7)
        for _ in range(500):
            n = rnd.randint(1, 12)
            lens = [rnd.choice([1, 1, 3, 4, 31, 32, 33, 64, 150, 400]) for _ in range(n)]
            order = list(range(n)); rnd.shuffle(order)
            first, row = [0] * n, 0
            for j in order:
                row += rnd.choice([0, 0, 1, 5]); first[j] = row; row += lens[j]
            t = row + rnd.choice([0, 7])
            seen = set()
            for s in range(n):
                rank = sum(1 for j in range(n) if lens[j] > 0 and first[j] < first[s])
                for c in range((lens[s] + 31) // 32):
                    slot = first[s] // 32 + rank + c
                    self.assertNotIn(slot, seen)
                    self.assertLess(slot, (t + 31) // 32 + n + 1)
                    seen.add(slot)

    @unittest.skipUnless(loom_compile(), "loom-compile not available")
    def test_compile_gfx1201(self):
        for spec in (False, True):
            _, t = self.kernels(spec)
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "t.loom"
                f.write_text("amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n" + t)
                r = subprocess.run([loom_compile(), str(f), "--format=amdgpu-hsaco", "--target=amdgpu:gfx1201", f"--output={d}/o.hsaco"], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[:2000])


if __name__ == "__main__":
    unittest.main()
