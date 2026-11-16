"""gfx12 fused GDN prefill (gen_gdn_fused.py): kernel text, gfx1201 compile, opt-in program wiring."""
import json, os, re, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "loom-port-llm" / "starter"))
import gen_gdn as GG  # noqa: E402
import gen_gdn_fused as GF  # noqa: E402

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
