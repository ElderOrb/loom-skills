#!/usr/bin/env python3
"""Generate the mv_q38 kbench files: Qwen3.8-27B's int8-activation decode matvecs (gen_llm._mv) at the model's real shapes on R9700.

usage: gen_mv_bench.py CFG.json OUTDIR SKILLS_STARTER SPEC...      CFG: the model's build cfg (with tensors); SPEC: fmt:K:n:TG:R:U[:sw]
A benchmark per SPEC: the kernel exactly as gen_llm._mv emits it for rows-per-workgroup R and U rows decoded per iteration
(U=1: the model's own kernel), weights beyond the 64 MB MALL (launches cycle over REPS distinct weight tensors, real-shape random bytes).
The bench's time is for REPS launches; bytes = REPS * weight bytes."""
import os, re, sys
cfg, out, starter = sys.argv[1:4]
os.environ["LLM_CONFIG"] = cfg
sys.path.insert(0, starter)
import gen_llm as G

HERE = os.path.dirname(os.path.abspath(__file__))
RB = {"q4_k": 144 / 256, "q6_k": 210 / 256, "q8_0": 34 / 32, "q5_k": 176 / 256}
cases = [open(os.path.join(HERE, "helpers.loom")).read(), G.ICODES]
for spec in sys.argv[4:]:
    f = spec.split(":")
    fmt, K, n, TG, R, U = f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])
    sw = len(f) > 6 and f[6] == "sw"
    name = f"k_{fmt}_k{K}_n{n}_t{TG}_r{R}_u{U}" + ("_sw" if sw else "")
    txt = G._mv(fmt, K, R, TG, swiglu=sw, suffix=f"_{name}", iq=True, unroll=U)
    old = re.search(r'kernel\.def export\("([^"]+)"\)', txt).group(1)
    txt = txt.replace(old, name).replace("kernel.def export(", "kernel.def target(@rdna4) export(", 1)
    if txt.count("kernel.def ") != 1: raise SystemExit("one kernel per spec")
    wb = int(n * K * RB[fmt]); nw = 2 if sw else 1
    xb = TG * K + TG * (K // 32) * 8
    # bytes per weight: reps keep the working set far beyond the MALL
    reps = max(2, -(-160_000_000 // (wb * nw)))
    decl = "".join(f"  %w{r}_{w} = check.generate.random.uniform seed(%seed) range(-100 to 100) : tensor<{wb}xi8>\n" for r in range(reps) for w in range(nw))
    launches = ""
    for r in range(reps):
        ws = ", ".join(f"%w{r}_{w}" for w in range(nw)); wt = ", ".join([f"tensor<{wb}xi8>"] * nw)
        launches += f"  kernel.launch @{name}[%n, %tk](%n, %tk, %acc, %x, {ws}, %y) : [index, index](index, index, i32, tensor<{xb}xi8>, {wt}, tensor<{TG * n}xf32>)\n"
    cases.append(txt)
    cases.append(f"""check.case public @{name}_case {{
  %seed = check.literal value(12345) : i64
  %n = check.literal value({n}) : index
  %tk = check.literal value({TG}) : index
  %acc = check.literal value(0) : i32
  %x = check.generate.random.uniform seed(%seed) range(-100 to 100) : tensor<{xb}xi8>
{decl}  %y = check.generate.fill value(0.0) : tensor<{TG * n}xf32>
{launches}  check.return
}}
check.benchmark<@{name}_case> @{name}_bm
""")
open(os.path.join(out, "bench.loom"), "w").write("amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n\n" + "\n".join(cases))
print("wrote", (len(cases) - 2) // 2, "benchmarks")
