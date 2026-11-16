# Model: amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom

`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (AWQ-calibrated OCP MXFP4, Quark export) served from its safetensors file: 12.9 GB of MXFP4 linears, bf16 embedding and head (5.1 GB), a bf16
MTP block (0.85 GB, the default drafter), a bf16 vision tower (0.92 GB, image input: read from the checkpoint, see facet/safetensors "Image tower"). Weights 19.8 GB: on the R9700 (32 GB) the served context and pool are limited by the weights plus the
verify scratch of the MTP drafter; the measured serving settings (`exec.*`, devices radeon-r9700) are written to `scope.json` when a DevLab run has measured them.
Source, name map and producer: loom-kb `facet/safetensors`, `facet/quark`, `quant/mxfp4s`.

MTP proposals (depth 3) read load-time copies, never the bf16 tensors: the NextN block as Q8_0 (`draft.propose.block`, 0.85 -> 0.45 GB) and a Q4_0 copy of head rows [0, 49152) + the added-token rows (`draft.propose.head_rows`, 0.15 GB instead of the 2.54 GB bf16 head); `llm_advance` / `llm_verify` keep the checkpoint's tensors (byte-identical programs with the choices off). R9700: decode 53.1 -> 69.2 tok/s, decode_2k 52.3 -> 68.8, acceptance .50 -> .46. The head copy is staged through four load-time lanes of the staged rows (4 x N x 10 KB of device budget, hrx-loom): more rows buy acceptance (N 32768 .41, 49152 .46, 81920 .49; decode 41 / 69 / 70) but cost served context (N = 81920 does not serve 81920 context with the image component), so N = 49152 and context 98304. Facet notes: `facet/safetensors`.

