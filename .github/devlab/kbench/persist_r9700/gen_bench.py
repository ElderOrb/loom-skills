#!/usr/bin/env python3
"""usage: gen_bench.py ROOT OUTDIR [VARIANT ...]   VARIANT: base (bench.loom) or itN (bench_itN.loom).

Persistent-kernel feasibility microbenchmark (campaign mxfp4st, lane dpersist; timing only). A P-phase chain, every phase = a streaming
read of IT x 512 KB of a (P x IT x 512 KB) slab by G = 128 workgroups of 256 lanes (a matvec-like weight stream), reduced into one i32 per lane
and stored; phase p+1 reads the stores of ANOTHER workgroup's phase p (a real cross-workgroup dependency). Three schedules of the same work:
  chain      P kernel launches (one phase each)                         = the current program's launch-per-op structure
  persist    one launch, P phases, grid barrier between phases          = ninfer's persistent decode kernel
  persist_pf as persist, with the next phase's first slab vector loaded before the barrier wait (next-phase weight prefetch)
The grid barrier is a device-scope atomic counter (view.atomic.rmw addi + acquire atomic-load spin, lane 0 of each workgroup, workgroup
barriers around it); the counter target is derived from the returned count so no reset between runs is needed. It relies on all G workgroups
being co-resident (G = 128 is far below the 64-CU x occupancy capacity); Loom has no cooperative-launch guarantee (notes/sdk_blockers.md)."""
import sys, os
G, P, LANES = 128, 64, 256
VARIANTS = {"it1": 1, "it8": 8, "it32": 32}

def kern(name, it, mode):
    """mode: 'one' (one phase, phase index = launch arg) | 'persist' | 'persist_pf'"""
    sweep = G * LANES                      # vector<4xi32> (16 B) per lane per iteration -> rows of the slab view
    rows_per_phase = sweep * it
    slab_rows = rows_per_phase * P
    args = "%p_in: index, " if mode == "one" else ""
    A = []
    a = A.append
    a(f'''kernel.def target(@rdna4) export("{name}") @{name}({"%p: index" if mode == "one" else ""}) {{
  %c1 = index.constant 1 : index
  %cg = index.constant {G} : index
  %cl = index.constant {LANES} : index
  kernel.launch.config workgroups(%cg, %c1, %c1) workgroup_size(%cl, %c1, %c1) : index
}} launch({args}%slab: buffer, %out: buffer, %cnt: buffer) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cg = index.constant {G} : index
  %cl = index.constant {LANES} : index
  %crows = index.constant {rows_per_phase} : index
  %csweep = index.constant {sweep} : index
  %slab_v = buffer.view %slab[%origin] : buffer -> view<{slab_rows}x4xi32>
  %out_v = buffer.view %out[%origin] : buffer -> view<2x{G * LANES}xi32>
  %cnt_v = buffer.view %cnt[%origin] : buffer -> view<1xi32>
  %wg0 = kernel.workgroup.id<x> : index
  %wg = index.assume %wg0 [range(%wg0, 0, {G - 1})] : index
  %lane0 = kernel.workitem.id<x> : index
  %lane = index.assume %lane0 [range(%lane0, 0, {LANES - 1})] : index
  %is0 = index.cmp eq, %lane, %c0 : index
  %wg_next0 = index.add %wg, %c1 : index
  %wg_next1 = index.rem %wg_next0, %cg : index
  %wg_next = index.assume %wg_next1 [range(%wg_next1, 0, {G - 1})] : index
  %g32 = scalar.constant {G} : i32
  %one32 = scalar.constant 1 : i32
  %zero32 = scalar.constant 0 : i32
  %my_col0 = index.mul %wg, %cl : index
  %my_col = index.add %my_col0, %lane : index
  %nb_col0 = index.mul %wg_next, %cl : index
  %nb_col = index.add %nb_col0, %lane : index
  %lane_off = index.mul %wg, %cl : index
  %lane_row = index.add %lane_off, %lane : index
''')
    def phase(pv, pm2, pf=False):
        """text of one phase body for phase index %{pv} (index) and parity %{pm2}; returns (text, prefetch...)"""
        s = f'''  %pbase_{pv}0 = index.mul %{pv}, %crows : index
  %pbase_{pv} = index.assume %pbase_{pv}0 [range(%pbase_{pv}0, 0, {slab_rows - 1})] : index
  %prev_par_{pv}0 = index.sub %c1, %{pm2} : index
  %prev_par_{pv} = index.assume %prev_par_{pv}0 [range(%prev_par_{pv}0, 0, 1)] : index
  %dep_{pv} = view.load %out_v[%prev_par_{pv}, %nb_col] : view<2x{G * LANES}xi32> -> i32
  %init_{pv} = {"scalar.addi %dep_" + pv + ", %pfv : i32" if pf else "scalar.addi %dep_" + pv + ", %zero32 : i32"}
  %acc_{pv} = scf.for %i = [%{"c1" if pf else "c0"} to %cit step %c1](%a = %init_{pv} : i32) -> (i32) {{
    %row0 = index.mul %i, %csweep : index
    %row1 = index.add %pbase_{pv}, %row0 : index
    %row2 = index.add %row1, %lane_row : index
    %row = index.assume %row2 [range(%row2, 0, {slab_rows - 1})] : index
    %v = vector.load %slab_v[%row, 0] : view<{slab_rows}x4xi32> -> vector<4xi32>
    %vs = vector.reduce<addi> %v, %a : vector<4xi32>, i32
    scf.yield %vs : i32
  }}
  view.store %acc_{pv}, %out_v[%{pm2}, %my_col] : i32, view<2x{G * LANES}xi32>
'''
        return s
    a(f"  %cit = index.constant {it} : index\n")
    if mode == "one":
        a("  %c2i = index.constant 2 : index\n  %p_arg = index.assume %p_in [range(%p_in, 0, " + str(P - 1) + ")] : index\n  %pm20 = index.rem %p_arg, %c2i : index\n  %pm2 = index.assume %pm20 [range(%pm20, 0, 1)] : index\n")
        a(phase("p_arg", "pm2").replace("%{pm2}", "%pm2"))
        a("  kernel.return\n}\n")
    else:
        a(f"  %cphases = index.constant {P} : index\n  %c2i = index.constant 2 : index\n")
        pfc = mode == "persist_pf"
        if pfc:
            a(f"  %cplast = index.constant {P - 1} : index\n  %pf_init0 = vector.load %slab_v[%lane_row, 0] : view<{slab_rows}x4xi32> -> vector<4xi32>\n  %pf_init = vector.reduce<addi> %pf_init0, %zero32 : vector<4xi32>, i32\n")
        a(f'''  {"%pf_end = " if pfc else ""}scf.for %pp0 = [%c0 to %cphases step %c1]({"%pfv = %pf_init : i32" if pfc else ""}){" -> (i32)" if pfc else ""} {{
    %pp = index.assume %pp0 [range(%pp0, 0, {P - 1})] : index
    %pm20 = index.rem %pp, %c2i : index
    %pm2 = index.assume %pm20 [range(%pm20, 0, 1)] : index
''')
        a(phase("pp", "pm2", pf=(mode == "persist_pf")))
        if mode == "persist_pf":   # next phase's first slab vector, loaded before the barrier wait (the weight prefetch of ninfer's persistent kernel)
            a(f'''    %pn0 = index.add %pp, %c1 : index
    %pn1 = index.min %pn0, %cplast : index
    %pn2 = index.mul %pn1, %crows : index
    %pn3 = index.add %pn2, %lane_row : index
    %pn = index.assume %pn3 [range(%pn3, 0, {slab_rows - 1})] : index
    %vn = vector.load %slab_v[%pn, 0] : view<{slab_rows}x4xi32> -> vector<4xi32>
    %pfn = vector.reduce<addi> %vn, %zero32 : vector<4xi32>, i32
''')
        # grid barrier: all stores of this phase visible to every workgroup before the next phase's loads
        a(f'''    buffer.fence scope(device) ordering(release)
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.if %is0 {{
      %old = view.atomic.rmw<addi> %one32, %cnt_v[0] {{ordering = acq_rel, scope = device}} : i32, view<1xi32> -> i32
      %gen = scalar.divui %old, %g32 : i32
      %gen1 = scalar.addi %gen, %one32 : i32
      %target = scalar.muli %gen1, %g32 : i32
      %t, %v2 = scf.while(%b = %old : i32) -> (i32, i32) {{
        %cur = view.atomic.load %cnt_v[0] {{ordering = acquire, scope = device}} : view<1xi32> -> i32
        %lt = scalar.cmpi slt, %cur, %target : i32
        scf.condition %lt, %b, %cur : i1, i32, i32
      }} do(%pw: i32, %qw: i32) {{
        scf.yield %pw : i32
      }}
      scf.yield
    }}
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    buffer.fence scope(device) ordering(acquire)
''')
        a(f"    scf.yield{' %pfn : i32' if pfc else ''}\n  }}\n  kernel.return\n}}\n")
    return "".join(A)

def case(name, mode, it):
    sweep = G * LANES; slab_rows = sweep * it * P
    if mode == "one":
        launches = "".join(f"  %p{i} = check.literal value({i}) : index\n  kernel.launch @{name}[%p{i}](%p{i}, %slab, %out, %cnt) : [index](index, tensor<{slab_rows}x4xi32>, tensor<2x{G * LANES}xi32>, tensor<1xi32>)\n" for i in range(P))
    else:
        launches = f"  kernel.launch @{name}[](%slab, %out, %cnt) : [](tensor<{slab_rows}x4xi32>, tensor<2x{G * LANES}xi32>, tensor<1xi32>)\n"
    return f'''check.case public @{name}_case {{
  %slab = check.generate.fill value(1) : tensor<{slab_rows}x4xi32>
  %out = check.generate.fill value(0) : tensor<2x{G * LANES}xi32>
  %cnt = check.generate.fill value(0) : tensor<1xi32>
{launches}  check.return
}}
check.benchmark<@{name}_case> @{name}_bm
'''

def bench(it):
    h = "target.decl @device\namdgpu.target<gfx12-generic> @rdna4\n"
    t = h
    for name, mode in (("k_chain", "one"), ("k_persist", "persist"), ("k_persist_pf", "persist_pf")):
        t += kern(name, it, mode) + case(name, mode, it)
    return t

def tests(outdir):
    """test.loom: both schedules against the closed form (slab = 1: phase p stores 4*IT*(p+1) in every lane); the persistent kernel is the
    grid-barrier correctness test (any lost wake-up / stale read changes a value or hangs the job)"""
    import numpy as np
    it = 2
    sweep = G * LANES; slab_rows = sweep * it * P
    exp = np.zeros((2, G * LANES), dtype=np.int32)
    for p in range(P):
        exp[p % 2, :] = 4 * it * (p + 1)
    np.save(os.path.join(outdir, "persist_expected.npy"), exp)
    t = "target.decl @device\namdgpu.target<gfx12-generic> @rdna4\n"
    for name, mode in (("k_chain", "one"), ("k_persist", "persist"), ("k_persist_pf", "persist_pf")):
        c = case(name, mode, it).replace("  check.return", f'  %e = check.file.read.npy path("persist_expected.npy") : tensor<2x{G * LANES}xi32>\n  check.expect.equal actual(%out) expected(%e) : tensor<2x{G * LANES}xi32>\n  check.return').replace("check.benchmark", "// check.benchmark")
        t += kern(name, it, mode) + c
    open(os.path.join(outdir, "test.loom"), "w").write(t)

def main():
    root, out = sys.argv[1], sys.argv[2]
    vs = sys.argv[3:] or ["base"]
    for v in vs:
        it = VARIANTS["it8" if v == "base" else v]
        fn = "bench.loom" if v == "base" else f"bench_{v}.loom"
        open(os.path.join(out, fn), "w").write(bench(it))
        print("wrote", fn)
    if "base" in vs:
        tests(out)

main()
