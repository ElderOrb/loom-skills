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

PLACEHOLDER (filled by the port PR): exact file names, the qwen35 table of per-tensor decisions (norm offset, A_log, conv1d, V-head order, q/gate
split, MTP names, mrope), `inspect_safetensors.py` options, and the vision-tower mapping or its gap.
