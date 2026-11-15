# vLLM as the gates' reference for safetensors packages

A package whose `llm_config.json` has `source.kind == "safetensors"` (for example a Quark MXFP4 checkpoint) has no GGUF, so
`gates.sh` uses vLLM ROCm serving the same checkpoint as the reference engine. Nothing else about the gates changes: the thresholds, the
near-tie rules and `compare.py`'s verdict are the same, and a failure that is only the reference's own numerics is shown as a failure on the
card with the evidence, never loosened.

- **Adapter**: `scripts/vllm_shim.py` is a llama-server look-alike (`/health`, `/props`, `/completion` with token ids in, greedy tokens + top-k
  log-probs out; `prompt_string` + `multimodal_data` for the vision gate) on top of vLLM's offline engine. Gates need only a URL.
  `scripts/vllm_server.sh` installs the pinned wheel (uv, python 3.12, `https://wheels.vllm.ai/rocm/VERSION/VARIANT`) and starts / stops it.
- **Settings**: bf16, greedy (`temperature 0`), seed 0, prefix caching off (every request is evaluated in full), one request at a time, the
  checkpoint's own quantization config (Quark MXFP4 is W4A4: vLLM emulates the fp4 activation quantization where the GPU has no native
  support), `max_model_len` = the package context + 1024 (at most 66560). `VLLM_VERSION` / `VLLM_ROCM_VARIANT` / `VLLM_EXTRA_ENV` select the build.
- **Cache**: references are cached like llama.cpp's (`llama_cache/`), keyed by `source.repo@revision`, the vLLM version (`LLAMA_CACHE_BUILD=vllm-VERSION`)
  and the request, so they do not depend on the machine running Loom: a gate job on another runner can restore the `llama_cache.tgz` a
  previous job uploaded.
- **Card statement**: `bench/reference.json` (from `/props`: engine, version, checkpoint, revision, dtype, quantization) is what the card cites.
- **No micro-batch view**: llama.cpp's slot-cache decode-kernel view (`_micro_batch`) has no vLLM equivalent; `llama_views` leaves it out and the
  near-tie rule uses the single-batch view.
- **Fallback** (vLLM cannot run on any runner): transformers + amd-quark on CPU behind the same `/completion` shim (a second engine class in
  `vllm_shim.py`).
