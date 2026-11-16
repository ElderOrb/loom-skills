# RDNA4 family (gfx12-generic, `@rdna4`)

Wave 32, WMMA 16x16 fragments with 8-wide operand vectors (`vector<8xf16>`: RDNA4 halves the fragment registers and doubles the dense
f16 rate per CU). A kernel with a gfx12 version `K` -> `K_rdna4` selects it per device (`dual_target.py`). The twins are text
rewrites of the RDNA3 kernels (`gen_gemm_rdna4.twin()`, `flash2_rdna4.to_rdna4()`, MoE inline in `gen_moe.py`); tuning that only
exists here: `flash2_b2`, `trim`, `vpack`, `prefetch`, `f16o_fuse`. Devices: `radeon-r9700` (gfx1201).

Data: `family.json`.

Radeon RX 9070 XT (Navi 48, gfx1201, 16 GB; not a registry device: no runner, no card GPU of its own). The R9700's packages run
unchanged (same chip, same programs); only the context that fits differs: `fleet/devices/rx9070xt.json`, by `loom-benchmark`
`test_9070xt.py` (first sweep 2026-11-16, hrx-loom 0.1.0, a desktop machine; contexts provisional under SDK 0.2.0's memory sizing).
Serving it (default preset, elastic): `--memory_bytes=15000000000` (17.1 GB board, ~1.5 GB held by a desktop session),
`--state_slab_bytes=4mib` (the default 64 MiB elastic state slab is committed per KV plane: 82 planes = 5.5 GB before the first
request, Qwen3-8B's context 7k instead of 32k), `--parameter_mode=preload` where `ulimit -l` is the stock 8 MB (the default
file mode pins io_uring buffers: "buffer pin preparation failed (12)", the model never loads). The R9700's `-r9700` preset with
`--pool_backing=fixed` commits a 32 GB-sized pool and fails here (RESOURCE_EXHAUSTED). gpt-oss-20b / LFM2.5-24B: the default prefill
chunk's workspace beside the weights exceeds 15 GB ("model parameters and invocation working set exceed the serving memory limit"):
a deployment document with `prefill_capacity` 1024 (gpt-oss-20b; LFM2.5-24B also needs 256 and `--weight_slab_bytes=64mib`, then serves
3k: marginal). These are the only per-model settings; on the R9700, elastic backing costs up to 28 % of long prefill (PR #1013), so
the R9700 keeps `--pool_backing=fixed` and the two GPUs need different serve settings.

