# Facet safetensors: porting a model from a checkpoint (recipe)

## Another MXFP4 safetensors model: what is configuration, what is code

Start: `python3 "${LOOM_SKILL_DIR}/../loom-port-llm/starter/inspect_safetensors.py" REPO_OR_DIR -o W/llm_config.json` (a hub repo is read with range
requests: header, config.json, tokenizer files; no weights), then route again with that config so the producer and quant nodes load.

| Step | Same architecture family as a ported model | New family |
|---|---|---|
| Name map | configuration: reuse `hf_names/<family>.json`; check every name resolves (the inspector lists unmapped tensors) | configuration: a new data file |
| Producer convention (Quark scale suffixes, exclude list) | configuration: read from the checkpoint's quantization config | code only for a producer the starter does not know |
| Quant format (`mxfp4s`) and its kernels | none: the format is registered, kernels are choice-selected | none |
| Geometry, chat template, tokenizer | `inspect_safetensors.py` output; run the cfg review (topic `cfg-review`) | same, plus `new-arch` if the architecture is new |
| Semantic differences of the framework layout (norm offset, A_log, conv1d, head order) | configuration: the family map's per-tensor entries | the entries are new; a layout no kernel reads natively needs a prepared copy |
| Package build | `build.py` from the config; `model.json` declares `format: safetensors` | same |
| Gates | reference = vLLM ROCm (topic `benchmark`) | same |

A checkpoint whose quantized tensors use another element order or scale type than the verified one is a new format key, not a flag on `mxfp4s`.

## Dry run without a GPU (the generality check)

Route the config (`kb.py route --config W/llm_config.json --topic porting`), build the package on CPU, and compile the programs for gfx1201 and gfx1151
with the SDK. A second Quark MXFP4 checkpoint should need a name map entry at most; anything more is a finding for this node.

PLACEHOLDER (filled by the port PR): the exact inspector and build.py command lines, the options that differ from the GGUF flow, the dry-run log of the
first model.
