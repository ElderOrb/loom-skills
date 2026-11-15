# MXFP4, split planes (`mxfp4s`)

OCP MX FP4 as a Quark / vLLM safetensors checkpoint stores it: every quantized linear `<name>.weight` is two tensors,
a **nibble plane** (U8 `[rows, K/2]`, two E2M1 codes per byte) and a **scale plane** (`<name>.weight_scale`, U8 `[rows, K/32]`, one
E8M0 per 32 values). Loom reads both in place (a `command.parameter` for each; no repack, no copy), so a call site carries two views
(`<weight>` and `<weight>_sc`) and the kernels take two operands. 32 values per block, 16 + 1 bytes per block (4.25 bits per weight).

- Element order: element `2j` is the low nibble of byte `j`, `2j+1` the high one (vLLM `mxfp4_w4a8_gemv_rdna3.cu`: "low nibble = even k";
  not ggml's `MXFP4`, whose block holds value `j` in the low and `j+16` in the high nibble). Verified on the real AMD checkpoint:
  the dequantized rows correlate 0.990-0.994 with the bf16 base model's rows, 0.06 for ggml's order, 0.00 for swapped nibbles.
- Value: `E2M1(code) * 2^(e - 127)`, codes `{0, .5, 1, 1.5, 2, 3, 4, 6}`, bit 3 = sign. Exact in f16 and bf16.
- Quark's `pack_method: reorder` is an int4 packing detail; the fp4 planes are in element order. The producer quantizes activations
  too (dynamic MXFP4, per 32 values): Loom keeps them f32 / f16 / int8 (more precise), which the gates against vLLM measure.
- Kernels: `g_mv_mxfp4s_k<K>[_t4][_r<R>]` (f32 activations, the decode matvecs through `@g_mxfp4s_values`) and `g_gemm_mxfp4s_store[_m128]`
  (the 64-row f16 WMMA GEMM; gate and up are two GEMMs and `g_silu_mul`). Fast kernels (int8 activations, fp8 WMMA) are a later choice.
- Source side: `starter/inspect_safetensors.py`, the per-family name maps `starter/hf_names/<family>.json` and the producer convention
  `starter/hf_names/quark.json`; reading and numpy references: `starter/st_tensors.py`.

Data: `format.json`; `starter/formats.py` stays the code source and a test keeps them equal.
