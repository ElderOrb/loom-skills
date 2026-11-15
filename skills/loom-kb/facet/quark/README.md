# Facet quark: checkpoints produced by AMD Quark

Selected by the llm_config key `quark` (the checkpoint's `quantization_config`, copied in by `inspect_safetensors.py`). Read together with
`facet/safetensors` (the container) and `quant/mxfp4s` (the tensor format).

## What the producer writes (config keys)

- `quant_method: "quark"`; `export.pack_method` ("reorder" in the AWQ MXFP4 export) says how the nibbles are ordered inside a byte; the verified element
  order is in `quant/mxfp4s`.
- `global_quant_config.weight`: fp4 e2m1, per_group, group_size 32, scale e8m0 (`scale_calculation_mode: even`, round half_even);
  `global_quant_config.input_tensors`: fp4 dynamic per_group 32. The producer's semantics are therefore **W4A4**; vLLM on a GPU without native fp4 emulates
  the activation quantization (quantize-dequantize), AMD-class hardware with fp4 matrix cores runs it natively.
- `exclude`: the module names that stay bf16 (here: the vision tower, the MTP module, lm_head, embeddings; linear attention gates and norms are not linears).
  Every other linear is quantized, including attention q/k/v/o and gated delta-net in_proj_a/b. The exclude list, not the tensor dtype alone, is the
  authority for which tensors are bf16.
- `layer_quant_config` is empty for the checkpoints seen; a non-empty one (per-layer override) is a finding.
- `algo_config`: null in the files seen; the calibration (AWQ) happened before export and shows only in the values.

## Tensor naming

A quantized linear `X` has `X.weight` (U8, `[rows, K/2]`, two e2m1 nibbles per byte) and `X.weight_scale` (U8 e8m0, `[rows, K/32]`). A bf16 tensor has
only `X.weight`. The inspector derives the format from the U8 dtype plus the scale tensor, then cross-checks it against `exclude`.

## What it means for Loom

- AWQ calibration: the scales were chosen with activation statistics, which is why this checkpoint is the reference quality (an uncalibrated round-to-nearest
  MXFP4 GGUF of the same model is lower quality). The weights are used exactly as stored, never requantized.
- Loom does not quantize activations to fp4: matvec activations are int8/f16, prefill activations may be fp8 e4m3 (`quant/mxfp4s`). The golden reference
  emulates A4, so exact token agreement is not expected on near-ties (see `facet/safetensors` benchmark task file).

## Known checkpoints

- amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (qwen35 hybrid, 19.8 GB, one safetensors file, repo sha 93245f7): MXFP4 linears 12.9 GB, bf16 embed and lm_head 5.1 GB,
  bf16 MTP 0.85 GB, bf16 vision 0.92 GB. Used by Paiton, Radiance and saber.

PLACEHOLDER (filled by the port PR): the verified meaning of `reorder`, scale-tensor edge cases (all-zero blocks, e8m0 255), other Quark exports seen (MoE
experts, fused qkv), and the producer module's name.
