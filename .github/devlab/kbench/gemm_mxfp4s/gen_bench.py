#!/usr/bin/env python3
"""Generate the gemm_mxfp4s kbench suite: the gfx12 MXFP4 split-plane fp8-WMMA prefill GEMMs (gen_mxfp4s.py) at the Qwen3.8-27B shapes.

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   ROOT: the loom-skills checkout; VARIANT: base (file bench.loom), or a name of VARIANTS (bench_<name>.loom).
Writes bench*.loom (one benchmark k_<mode>_<K>x<N>_<rows> per shape: random planar weights, timing only) and test.loom (check.cases against a numpy
float64 reference on the exactly dequantized weights and the e4m3 activations given; run by devlab_kbench.sh's "tests" step with loom-test)."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-port-llm", "starter"))
import numpy as np
import gen_mxfp4s as G

HEAD = "amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n"
ROWS = 2048
SHAPES = [("swiglu", 5120, 17408), ("store", 17408, 5120), ("store", 5120, 12288), ("store", 5120, 10240), ("store", 5120, 6144), ("store", 6144, 5120)]
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64)
VARIANTS = {}   # name -> function(kernel text, mode) -> kernel text (levers, filled in as they are tried)


def kernel(mode, kn, variant="", order="pairs"):
    t = G.gemm_mxfp4s_f8_m256(mode, order=order)
    old = f"g_gemm_mxfp4s_{mode}_f8_m256_rdna4"
    t = t.replace(f'export("{old}") @{old}(', f'export("{kn}") @{kn}(')
    return VARIANTS[variant](t, mode) if variant else t


def build(variant=""):
    out = []
    for mode, K, N in SHAPES:
        kn = f"k_{mode}_{K}x{N}_{ROWS}"
        out.append(kernel(mode, kn, variant))
        planes = ["w"] + (["u"] if mode == "swiglu" else [])
        decl = "".join(f"  %{p} = check.generate.random.uniform seed(%seed) range(-128 to 127) : tensor<{N * K // 2}xi8>\n"
                       f"  %{p}s = check.generate.random.uniform seed(%seed) range(116 to 126) : tensor<{N * K // 32}xi8>\n" for p in planes)
        args = ", ".join(f"%{p}, %{p}s" for p in planes)
        tys = ", ".join(f"tensor<{N * K // 2}xi8>, tensor<{N * K // 32}xi8>" for _ in planes)
        out.append(f"""check.case public @{kn}_case {{
  %seed = check.literal value(12345) : i64
  %rows = check.literal value({ROWS}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({N}) : index
  %x = check.generate.random.uniform seed(%seed) range(-100 to 100) : tensor<{ROWS}x{K}xi8>
  %s = check.generate.fill value(1.0) : tensor<{ROWS}x2xf32>
{decl}  %y = check.generate.fill value(0.0) : tensor<{ROWS}x{N}xf32>
  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, %x, %s, {args}, %y) : [index, index](index, index, index, tensor<{ROWS}x{K}xi8>, tensor<{ROWS}x2xf32>, {tys}, tensor<{ROWS}x{N}xf32>)
  check.return
}}
check.benchmark<@{kn}_case> @{kn}_bm
""")
    return out


# -- correctness cases --------------------------------------------------------------------------------------------------
def e4m3_round(x):
    x = np.asarray(x, np.float64); a = np.minimum(np.abs(x), 448.0)
    e = np.maximum(np.floor(np.log2(np.maximum(a, 2.0 ** -20))), -6)
    q = 2.0 ** (e - 3)
    return np.sign(x) * np.minimum(np.rint(a / q) * q, 448.0)


def e4m3_bytes(v):
    v = np.asarray(v, np.float64); a = np.abs(v)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -20))).astype(np.int64)
    sub = a < 2.0 ** -6
    ef = np.where(sub, 0, e + 7); mant = np.where(sub, np.rint(a / 2.0 ** -9), np.rint((a / 2.0 ** e - 1) * 8)).astype(np.int64)
    b = np.where(a == 0, 0, (ef << 3) | mant) | np.where(v < 0, 0x80, 0)
    return b.astype(np.uint8).view(np.int8)


def dequant(nib, sc, order):
    """nib [n, K/2] U8, sc [n, K/32] U8 -> float64 [n, K]: element values fp4 * 2^(e - 127), the block's elements in `order`."""
    n, kh = nib.shape
    lo, hi = FP4[nib & 7] * np.where(nib & 8, -1.0, 1.0), FP4[(nib >> 4) & 7] * np.where(nib & 128, -1.0, 1.0)
    lo, hi = lo.reshape(n, -1, 16), hi.reshape(n, -1, 16)
    blk = np.stack([lo, hi], 3).reshape(n, -1, 32) if order == "pairs" else np.concatenate([lo, hi], 2)
    return (blk * 2.0 ** (sc.astype(np.float64)[:, :, None] - 127)).reshape(n, -1)


def save(d, name, a):
    np.save(os.path.join(d, name + ".npy"), a)


def tests(outdir):
    rng = np.random.default_rng(7)
    out = []
    for order in ("pairs", "split16"):
        for mode, K, n in (("store", 5120, 1024), ("swiglu", 5120, 512), ("store", 17408, 256)):
            rows = 512
            kn = f"t_{mode}_{order}_{K}"
            nib = [rng.integers(0, 256, (n, K // 2)).astype(np.uint8) for _ in range(2 if mode == "swiglu" else 1)]
            sc = [rng.integers(108, 127, (n, K // 32)).astype(np.uint8) for _ in nib]
            W = [dequant(a, b, order) for a, b in zip(nib, sc)]
            xf = e4m3_round(np.clip(rng.standard_normal((rows, K)) * 30.0, -440, 440))
            inv = rng.uniform(0.25, 2.0, rows).astype(np.float32)
            s = np.stack([np.ones(rows, np.float32), inv * np.float32(2.0 ** -8)], 1)
            g = (xf @ W[0].T) * inv[:, None].astype(np.float64)
            if mode == "swiglu":
                u = (xf @ W[1].T) * inv[:, None].astype(np.float64)
                y = g / (1 + np.exp(np.clip(-g, -80, 80))) * u
            else:
                y = g
            save(outdir, kn + "_x", e4m3_bytes(xf)); save(outdir, kn + "_s", s); save(outdir, kn + "_y", y.astype(np.float32))
            decl, args, tys = "", [], []
            for i, (a, b) in enumerate(zip(nib, sc)):
                save(outdir, f"{kn}_w{i}", a.reshape(-1).view(np.int8)); save(outdir, f"{kn}_ws{i}", b.reshape(-1).view(np.int8))
                decl += (f'  %w{i} = check.file.read.npy path("{kn}_w{i}.npy") : tensor<{a.size}xi8>\n'
                         f'  %ws{i} = check.file.read.npy path("{kn}_ws{i}.npy") : tensor<{b.size}xi8>\n')
                args += [f"%w{i}", f"%ws{i}"]; tys += [f"tensor<{a.size}xi8>", f"tensor<{b.size}xi8>"]
            tol = 1.0e-3 * float(np.abs(y).std() + 1e-6)
            out.append(kernel(mode, kn, order=order))
            out.append(f"""check.case public @{kn}_case {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({n}) : index
  %x = check.file.read.npy path("{kn}_x.npy") : tensor<{rows}x{K}xi8>
  %s = check.file.read.npy path("{kn}_s.npy") : tensor<{rows}x2xf32>
{decl}  %e = check.file.read.npy path("{kn}_y.npy") : tensor<{rows}x{n}xf32>
  %y = check.generate.fill value(0.0) : tensor<{rows}x{n}xf32>
  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, %x, %s, {", ".join(args)}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xi8>, tensor<{rows}x2xf32>, {", ".join(tys)}, tensor<{rows}x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol({tol:.3e}) rtol(2.0e-3) nan(different) : tensor<{rows}x{n}xf32>
  check.return
}}
""")
    return out


if __name__ == "__main__" and len(sys.argv) > 2:
    d = sys.argv[2]
    os.makedirs(d, exist_ok=True)
    for v in (sys.argv[3:] or ["base"]):
        v = "" if v == "base" else v
        open(os.path.join(d, "bench%s.loom" % ("_" + v if v else "")), "w").write(HEAD + "\n".join(build(v)))
    open(os.path.join(d, "test.loom"), "w").write(HEAD + "\n".join(tests(d)))
