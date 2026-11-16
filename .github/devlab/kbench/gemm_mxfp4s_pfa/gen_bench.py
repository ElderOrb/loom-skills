#!/usr/bin/env python3
"""Generate the gemm_mxfp4s_pfa kbench suite (lane pfa): the producer-free MXFP4 fp8-WMMA prefill GEMM (gen_mxfp4s_pfa.py: the GEMM's prologue makes its own
fp8 activations from the f32 rows) against the shipped path (gen_mxfp4s.gemm_mxfp4s_f8_m256 on producer-made fp8 operands) and the producers themselves, at the
Qwen3.8-27B shapes, in one job so the sums compare: pfa GEMM vs (a_direct GEMM + producer).

usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   VARIANT: base (bench.loom), or `r1024` (the card's prefill chunk; bench_r1024.loom).
Benchmarks per shape (mode, K, N) at ROWS rows: k_base_*: shipped GEMM on fp8 operands; k_pfa_*: producer-free GEMM on f32 rows (plain); k_pfn_*: producer-free
GEMM with the RMSNorm in the prologue (K 5120 shapes); k_prod_*: the producer twins (row scale of an f32 row -> fp8 + scale; post-add-norm). test.loom: float64
references (exact dequantized weights, the e4m3 quantization of the kernel's rows emulated) incl. dead rows and a fully dead row tile."""
import os, re, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-port-llm", "starter"))
import numpy as np
import gen_mxfp4s as G
import gen_mxfp4s_pfa as P
spec = importlib.util.spec_from_file_location("gb_mxfp4s", os.path.join(HERE, "..", "gemm_mxfp4s", "gen_bench.py"))
GB = importlib.util.module_from_spec(spec); spec.loader.exec_module(GB)   # e4m3_round, dequant, FP4 of the sibling suite

HEAD = "amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n"
SHAPES = [("swiglu", 5120, 17408), ("store", 17408, 5120), ("store", 5120, 12288), ("store", 5120, 10240), ("store", 5120, 6144), ("store", 6144, 5120)]
FIX = open(os.path.join(ROOT, "tests", "data", "f8_producers.loom")).read()


def fixture(name):
    a = FIX.index(f'kernel.def export("{name}")'); end = "  kernel.return\n}\n"
    return FIX[a:FIX.index(end, a) + len(end)]


def rename(t, old, new):
    return t.replace(f'export("{old}") @{old}(', f'export("{new}") @{new}(')


def build(rows):
    out = []
    info = f"  %info = check.generate.fill value(1) : tensor<{rows}x4xi32>\n"
    for mode, K, N in SHAPES:
        planes = ["w"] + (["u"] if mode == "swiglu" else [])
        decl = "".join(f"  %{p} = check.generate.random.uniform seed(%seed) range(-128 to 127) : tensor<{N * K // 2}xi8>\n"
                       f"  %{p}s = check.generate.random.uniform seed(%seed) range(116 to 126) : tensor<{N * K // 32}xi8>\n" for p in planes)
        args = ", ".join(f"%{p}, %{p}s" for p in planes)
        tys = ", ".join(f"tensor<{N * K // 2}xi8>, tensor<{N * K // 32}xi8>" for _ in planes)
        tag = f"{mode}_{K}x{N}_{rows}"
        # shipped GEMM on producer-made fp8 operands
        kb = f"k_base_{tag}"
        out.append(rename(G.gemm_mxfp4s_f8_m256(mode), f"g_gemm_mxfp4s_{mode}_f8_m256_rdna4", kb))
        out.append(f"""check.case public @{kb}_case {{
  %seed = check.literal value(12345) : i64
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({N}) : index
  %x = check.generate.random.uniform seed(%seed) range(-100 to 100) : tensor<{rows}x{K}xi8>
  %s = check.generate.fill value(1.0) : tensor<{rows}x2xf32>
{decl}  %y = check.generate.fill value(0.0) : tensor<{rows}x{N}xf32>
  kernel.launch @{kb}[%rows, %n](%rows, %k, %n, %x, %s, {args}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xi8>, tensor<{rows}x2xf32>, {tys}, tensor<{rows}x{N}xf32>)
  check.return
}}
check.benchmark<@{kb}_case> @{kb}_bm
""")
        for norm in (False, True):
            if norm and K != 5120:
                continue
            kp = f"k_{'pfn' if norm else 'pfa'}_{tag}"
            out.append(rename(P.gemm_mxfp4s_f8_pfa(mode, norm=norm), f"g_gemm_mxfp4s_{mode}_f8_pfa{'_norm' if norm else ''}_rdna4", kp))
            out.append(f"""check.case public @{kp}_case {{
  %seed = check.literal value(12345) : i64
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({N}) : index
  %x = check.generate.random.uniform seed(%seed) range(-4.0 to 4.0) : tensor<{rows}x{K}xf32>
  %nw = check.generate.random.uniform seed(%seed) range(0.5 to 1.5) : tensor<{K}xf32>
{info}{decl}  %y = check.generate.fill value(0.0) : tensor<{rows}x{N}xf32>
  kernel.launch @{kp}[%rows, %n](%rows, %k, %n, %x, %nw, %info, {args}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xf32>, tensor<{K}xf32>, tensor<{rows}x4xi32>, {tys}, tensor<{rows}x{N}xf32>)
  check.return
}}
check.benchmark<@{kp}_case> @{kp}_bm
""")
    # the producer twins, as the program launches them in a fp8 epoch (w16 = 0: no f16 operand copy)
    for K in (5120, 6144, 17408):
        kn = f"k_prod_rs_{K}_{rows}"
        t = G.f8_producer(fixture("g_row_scale_f16_live_lv"))
        t = rename(t, "g_row_scale_f16_live_lv_f8_rdna4", kn)
        out.append(t)
        out.append(f"""check.case public @{kn}_case {{
  %seed = check.literal value(12345) : i64
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %w16 = check.literal value(0) : index
  %x = check.generate.random.uniform seed(%seed) range(-4.0 to 4.0) : tensor<{rows}x{K}xf32>
{info}  %s = check.generate.fill value(0.0) : tensor<{rows}x2xf32>
  %x16 = check.generate.fill value(0.0) : tensor<{rows}x{K}xf16>
  %x8 = check.generate.fill value(0) : tensor<{rows}x{K}xi8>
  %s8 = check.generate.fill value(0.0) : tensor<{rows}x2xf32>
  kernel.launch @{kn}[%rows](%rows, %info, %k, %x, %s, %x16, %x8, %s8, %w16) : [index](index, tensor<{rows}x4xi32>, index, tensor<{rows}x{K}xf32>, tensor<{rows}x2xf32>, tensor<{rows}x{K}xf16>, tensor<{rows}x{K}xi8>, tensor<{rows}x2xf32>, index)
  check.return
}}
check.benchmark<@{kn}_case> @{kn}_bm
""")
    kn = f"k_prod_pan_5120_{rows}"
    t = rename(G.f8_producer(fixture("g_post_add_norm_rs_live_lv")), "g_post_add_norm_rs_live_lv_f8_rdna4", kn)
    out.append(t)
    out.append(f"""check.case public @{kn}_case {{
  %seed = check.literal value(12345) : i64
  %rows = check.literal value({rows}) : index
  %w16 = check.literal value(0) : index
  %y = check.generate.random.uniform seed(%seed) range(-4.0 to 4.0) : tensor<{rows}x5120xf32>
  %wp = check.generate.random.uniform seed(%seed) range(0.5 to 1.5) : tensor<5120xf32>
  %x = check.generate.random.uniform seed(%seed) range(-4.0 to 4.0) : tensor<{rows}x5120xf32>
  %wn = check.generate.random.uniform seed(%seed) range(0.5 to 1.5) : tensor<5120xf32>
{info}  %h = check.generate.fill value(0.0) : tensor<{rows}x5120xf32>
  %s = check.generate.fill value(0.0) : tensor<{rows}x2xf32>
  %h16 = check.generate.fill value(0.0) : tensor<{rows}x5120xf16>
  %h8 = check.generate.fill value(0) : tensor<{rows}x5120xi8>
  %s8 = check.generate.fill value(0.0) : tensor<{rows}x2xf32>
  kernel.launch @{kn}[%rows](%rows, %info, %y, %wp, %x, %wn, %h, %s, %h16, %h8, %s8, %w16) : [index](index, tensor<{rows}x4xi32>, tensor<{rows}x5120xf32>, tensor<5120xf32>, tensor<{rows}x5120xf32>, tensor<5120xf32>, tensor<{rows}x5120xf32>, tensor<{rows}x2xf32>, tensor<{rows}x5120xf16>, tensor<{rows}x5120xi8>, tensor<{rows}x2xf32>, index)
  check.return
}}
check.benchmark<@{kn}_case> @{kn}_bm
""")
    return out


def save(d, name, a):
    np.save(os.path.join(d, name + ".npy"), a)


def reference_rows(x, nw, norm, live):
    """the kernel's per-row arithmetic in float64: h, the power-of-two scale from the f32 exponent of max |h|, the e4m3 values and the row factor"""
    h = x * nw if norm else x
    if norm:
        rs = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(1) + 1.0e-6)
        h = h * rs[:, None]
    mx = np.abs(h).max(1)
    biased = np.floor(np.log2(np.maximum(mx, 1e-30))).astype(np.int64) + 127
    scale = np.where(live, 2.0 ** (261 - biased - 127.0), 0.0)
    q = GB.e4m3_round(h * scale[:, None])
    inv = np.where(live, 2.0 ** (biased - 142.0), 0.0)
    return q, inv


def tests(outdir):
    rng = np.random.default_rng(11)
    out = []
    rows = 768
    live = np.ones(rows, bool); live[256:512] = False; live[640:700] = False   # a dead row tile and a partly dead one
    info = np.zeros((rows, 4), np.int32); info[:, 2] = live.astype(np.int32)
    save(outdir, "pfa_info", info)
    for order in ("pairs", "split16"):
        for mode, K, n, norm in (("store", 5120, 1024, False), ("store", 5120, 1024, True), ("swiglu", 5120, 512, False), ("swiglu", 5120, 512, True), ("store", 17408, 256, False)):
            if order == "split16" and (norm or K != 5120):
                continue
            kn = f"t_pfa_{mode}_{order}_{K}{'_norm' if norm else ''}"
            nib = [rng.integers(0, 256, (n, K // 2)).astype(np.uint8) for _ in range(2 if mode == "swiglu" else 1)]
            sc = [rng.integers(108, 127, (n, K // 32)).astype(np.uint8) for _ in nib]
            W = [GB.dequant(a, b, order) for a, b in zip(nib, sc)]
            x = (rng.standard_normal((rows, K)) * rng.uniform(0.2, 6.0, (rows, 1))).astype(np.float32)
            nw = rng.uniform(0.5, 1.5, K).astype(np.float32)
            q, inv = reference_rows(x.astype(np.float64), nw.astype(np.float64), norm, live)
            g = (q @ W[0].T) * inv[:, None]
            if mode == "swiglu":
                u = (q @ W[1].T) * inv[:, None]
                y = g / (1 + np.exp(np.clip(-g, -80, 80))) * u
            else:
                y = g
            y = np.where(live[:, None], y, 0.0).astype(np.float32)
            save(outdir, kn + "_x", x); save(outdir, kn + "_nw", nw); save(outdir, kn + "_y", y)
            decl, args, tys = "", [], []
            for i, (a, b) in enumerate(zip(nib, sc)):
                save(outdir, f"{kn}_w{i}", a.reshape(-1).view(np.int8)); save(outdir, f"{kn}_ws{i}", b.reshape(-1).view(np.int8))
                decl += (f'  %w{i} = check.file.read.npy path("{kn}_w{i}.npy") : tensor<{a.size}xi8>\n'
                         f'  %ws{i} = check.file.read.npy path("{kn}_ws{i}.npy") : tensor<{b.size}xi8>\n')
                args += [f"%w{i}", f"%ws{i}"]; tys += [f"tensor<{a.size}xi8>", f"tensor<{b.size}xi8>"]
            tol = 3.0e-3 * float(np.abs(y).std() + 1e-6)
            out.append(rename(P.gemm_mxfp4s_f8_pfa(mode, order=order, norm=norm, kdim=K), f"g_gemm_mxfp4s_{mode}_f8_pfa{'_norm' if norm else ''}_rdna4", kn))
            out.append(f"""check.case public @{kn}_case {{
  %rows = check.literal value({rows}) : index
  %k = check.literal value({K}) : index
  %n = check.literal value({n}) : index
  %x = check.file.read.npy path("{kn}_x.npy") : tensor<{rows}x{K}xf32>
  %nw = check.file.read.npy path("{kn}_nw.npy") : tensor<{K}xf32>
  %info = check.file.read.npy path("pfa_info.npy") : tensor<{rows}x4xi32>
{decl}  %e = check.file.read.npy path("{kn}_y.npy") : tensor<{rows}x{n}xf32>
  %y = check.generate.fill value(0.0) : tensor<{rows}x{n}xf32>
  kernel.launch @{kn}[%rows, %n](%rows, %k, %n, %x, %nw, %info, {", ".join(args)}, %y) : [index, index](index, index, index, tensor<{rows}x{K}xf32>, tensor<{K}xf32>, tensor<{rows}x4xi32>, {", ".join(tys)}, tensor<{rows}x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol({tol:.3e}) rtol(3.0e-3) nan(different) : tensor<{rows}x{n}xf32>
  check.return
}}
""")
    return out


if __name__ == "__main__" and len(sys.argv) > 2:
    d = sys.argv[2]
    os.makedirs(d, exist_ok=True)
    for v in (sys.argv[3:] or ["base"]):
        rows = 2048
        if re.fullmatch(r"r\d+", v):
            rows = int(v[1:])
        elif v != "base":
            raise SystemExit("unknown variant " + v)
        open(os.path.join(d, "bench%s.loom" % ("_" + v if v != "base" else "")), "w").write(HEAD + "\n".join(build(rows)))
    text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", HEAD + "\n".join(tests(d)), flags=re.M)
    open(os.path.join(d, "test.loom"), "w").write(text)
