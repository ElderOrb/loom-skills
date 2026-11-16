# RDNA4 family (gfx12-generic, `@rdna4`)

Wave 32, WMMA 16x16 fragments with 8-wide operand vectors (`vector<8xf16>`: RDNA4 halves the fragment registers and doubles the dense
f16 rate per CU). A kernel with a gfx12 version `K` -> `K_rdna4` selects it per device (`dual_target.py`). The twins are text
rewrites of the RDNA3 kernels (`gen_gemm_rdna4.twin()`, `flash2_rdna4.to_rdna4()`, MoE inline in `gen_moe.py`); tuning that only
exists here: `flash2_b2`, `trim`, `vpack`, `prefetch`, `f16o_fuse`. Devices: `radeon-r9700` (gfx1201).

Data: `family.json`.

Profiling on gfx1201 (kbench run 43803638485, hrx-loom 0.1.0): per-dispatch device timestamps work (`profile_replay`, `--print_device_statistics`
with `dispatch-events`); hardware counters, counter-ranges, device metrics and executable traces are UNIMPLEMENTED (`profile.c:636`);
`--compile-report=details` works (`matrix_fp8/bf8_native_kind` unscaled, `matrix_fp4_native_kind` none: no native MXFP4 matrix op on gfx12).
Recipe: topic `optimize-loop`, "Profiling".
