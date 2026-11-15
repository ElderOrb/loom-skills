# Model: amd-Qwen3.8-27B-Quark-AWQ-MXFP4-loom

`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (AWQ-calibrated OCP MXFP4, Quark export) served from its safetensors file: 12.9 GB of MXFP4 linears, bf16 embedding and head (5.1 GB), a bf16
MTP block (0.85 GB, the default drafter), a bf16 vision tower (not wired yet). Weights 19.8 GB: on the R9700 (32 GB) the served context and pool are limited by the weights plus the
verify scratch of the MTP drafter; the measured serving settings (`exec.*`, devices radeon-r9700) are written to `scope.json` when a DevLab run has measured them.
Source, name map and producer: loom-kb `facet/safetensors`, `facet/quark`, `quant/mxfp4s`.
