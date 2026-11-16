# Facet: safetensors source (a package built from a Hugging Face checkpoint, no GGUF)

Selected by the llm_config key `source` (`{"kind": "safetensors", "repo", "revision", "files"}`, written by `inspect_safetensors.py`). A GGUF
config never carries `source`, so GGUF ports never load this node; a test (`tests/test_kb_source_nodes.py`) keeps that true. The producer node
loads by its own key (`facet/quark`) and the quantization format node by `formats` in the config or `--quant` (`facet/quark` also prints
`quant/mxfp4s`), so this node holds only what is true of reading any safetensors checkpoint.

## How the source differs from a GGUF port

- The weights are read in place: model.json declares the parameter file `{"path": "parameter/model.safetensors", "format": "safetensors"}` (a symlink
  into the HF cache is fine, never a copy) and `command.parameter` carries the checkpoint's own tensor key. Programs and the starter keep the internal
  (GGUF-style) tensor names; the per-family name map turns one into the other.
- There is no GGUF metadata: geometry comes from the checkpoint's `config.json` (text_config for a multimodal wrapper), the tokenizer from
  `tokenizer.json` in the same repo (the chat template from `tokenizer_config.json` / `chat_template.jinja`), the tensor table from the safetensors
  header (an 8-byte little-endian length, then JSON: dtype, shape, byte offsets per tensor). A header is readable with HTTP range reads, so a port
  starts without the download; fleet entries keep a header dump the way GGUF ones keep a sparse stub.
- Names are HF names (`model.language_model.layers.N.self_attn.q_proj.weight`), and the layouts are the framework's, not ggml's.

## Load-time prepared copies (the rule)

Where a tensor's HF semantics differ from what the generators assume (norm `(1 + w)` offsets, `A_log` stored as the log of the decay, conv1d weight
layout, V-head order of a grouped delta-net, a fused or split q/gate projection), either the kernel reads the tensor natively or the prepare program
writes one small derived copy at load time. A prepared copy is bf16 or f32 and small (norms, gates, conv taps, head tables). **A quantized tensor is
never requantized or re-rounded**: a prepared copy of quantized values would change the numbers the producer calibrated. If a layout needs a
different operand order, the kernel reads the stored order. Each tensor's choice (native or prepared) is recorded in the family name map next to its
pattern, and a prepared copy that feeds only a proposal (drafter head) is the only bf16 -> lower-precision copy allowed without an A/B and a
decision note.

## Family name maps

One data file per family (`starter/hf_names/<family>.json`: internal-name pattern -> HF-key pattern, plus the per-tensor preparation). A new family
of an already ported architecture is a new data file, not code; a producer convention (which tensors are quantized, which stay bf16, scale tensor
suffixes) is a separate module per producer (`facet/quark`).

Files: `starter/hf_names/qwen35.json` (the qwen35 family: Qwen3.5 / Qwen3.8 dense hybrids), `starter/hf_names/llama.json` (LlamaForCausalLM: dense GQA, plain RMSNorm, neox rope; a family map may omit `layer_types`, `gdn`, `mtp`, per-layer `linear_attention`), `starter/hf_names/quark.json` (producer), `starter/inspect_safetensors.py`
(`python3 inspect_safetensors.py REPO_OR_DIR OUT.json [--revision REV] [--checkpoint LOCAL.safetensors]`; `-o OUT.json` also works; a dump written by `fleet/st_stub.py` is a valid source),
`starter/st_tensors.py` (numpy reader and references for the kernel tests). The inspector prints the checkpoint tensors the map does not use and the vision tensors.

qwen35, per tensor (HF -> internal; "prep" is the load-time conversion bf16 -> f32 of the weight policy, `draft/prep_weights.py`):

| HF tensor | internal | decision |
|---|---|---|
| every RMSNorm weight (input / post_attention layernorm, q_norm, k_norm, final norm, MTP norms) | `attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`, `output_norm`, `nextn.*norm` | zero-centred `(1 + w)` (vLLM GemmaRMSNorm): prep `add1` |
| `linear_attn.norm.weight` | `ssm_norm.weight` | plain gated RMSNorm weight: prep `copy` |
| `linear_attn.A_log` | `ssm_a` | prep `negexp` (-exp(A_log), f32 `<afn>` exp) |
| `linear_attn.dt_bias`, `conv1d.weight` [C,1,K] | `ssm_dt.bias`, `ssm_conv1d.weight` [K,C] | prep `copy` (same element order) |
| value heads of in_proj_qkv V rows, in_proj_z, in_proj_a/b, out_proj columns, A_log, dt_bias, conv1d V channels | the same tensors | HF order is grouped (value head h reads key head h // (HV/HK)); llama.cpp reorders to tiled. Read natively: cfg `gdn.v_order: "grouped"`, `gen_gdn.kh_line()`; no weight is permuted |
| `self_attn.q_proj` [heads*2*head_dim] | `attn_q` | per head [query, gate]: the same packing as the GGUF (`attn_gate: fused_q`) |
| `in_proj_qkv` rows [q, k, v] | `attn_qkv` | the same as the GGUF (no qkvz interleave in Qwen3.5) |
| `mtp.fc`, `mtp.pre_fc_norm_embedding` / `_hidden`, `mtp.norm`, `mtp.layers.0.*` | `blk.L.nextn.eh_proj`, `enorm`, `hnorm`, `shared_head_norm`, `blk.L.*` | bf16 (excluded from quantization), MTP depth 3 as the GGUF card |
| `embed_tokens`, `lm_head` | `token_embd`, `output` | bf16, untied |
| rope | `rope_dims` 64 of 256, neox, theta 1e7 | mrope sections only matter for image positions |
| `rope` mrope_section [11, 11, 10] interleaved | cfg `mrope` sections [11, 11, 10, 0], interleaved | image positions (t, h, w): written with the vision block, as the GGUF's `rope.dimension_sections` |
| `model.visual.*` (27-block ViT, bf16) | cfg `vision` (the image component) | see "Image tower" below |

### Image tower (`model.visual.*`)

`inspect_safetensors.py` writes `cfg["vision"]` when the map has a `vision` block and the header has the prefix: the encoder config from `config.json` `vision_config`
(+ `preprocessor_config.json` mean / std), and the tensor table under the **mmproj converter's names** (`v.blk.N.attn_qkv.weight`, `mm.0.weight`, ...), each with its HF key, so the
encoder plan (`starter/vision/encoder.py`, family `qwen3vl_merger`) is shared with the GGUF path. The image programs read HF keys; the card's `parameters.vision` is the checkpoint file itself (no projector file).
`st_source.py` builds the block (stdlib only); `encoder.config_st()` turns it into the plan's config. Per tensor (map: `hf_names/qwen35.json` "vision"):

| HF tensor | decision |
|---|---|
| `blocks.N.attn.qkv`, `attn.proj`, `mlp.linear_fc1/2`, `merger.linear_fc1/2` weights | bf16 as stored (llama.cpp rounds them to f16; the file's own values are read, the kernels take bf16) |
| biases, `norm1/2`, `merger.norm`, `pos_embed.weight` | prepared f32 copies (bf16 -> f32 at load, op `copy`; policy `vision_weights.loom`, bound by `vision_prepare.loom`): the encoder kernels read f32 there (10 MB for the position table) |
| `patch_embed.proj.weight` [E, 3, 2, P, P] (conv3d) | read natively (kernel layout `hf3d`: sum of the two temporal taps, i.e. the converter's `patch_embd.weight` + `.1`, one tensor) |
| `deepstack_visual_indexes` | `deepstack_merger_list.K.*` -> `v.deepstack.I.*` (empty for this model) |

Preprocessing (resize, 2x2 token blocks, `<|vision_start|>` / `<|vision_end|>`, token limits 8..4096) is the llama.cpp / mtmd rule of the family, as for the GGUF package; the HF processor's `size` (min 65536 px, max 16777216 px) differs only for images under 256x256.
Checks: `tests/test_safetensors_vision.py` (names and shapes equal the mmproj's; on the real files the tensors equal the mmproj values to 1e-6 and `ref_vit.py` gives the same rows from either, cos > 0.99999), kernel case `vision/tests/test_embed.py` (hf3d), whole encoder on device `vision/test_encoder.py CFG.json CHECKPOINT H W`.

Test: `tests/test_safetensors_source.py`; the dequant of real rows against a scalar reimplementation of vLLM's decode runs when the checkpoint is in the HF cache.

