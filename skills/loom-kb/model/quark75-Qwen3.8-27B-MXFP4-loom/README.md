# Model: quark75-Qwen3.8-27B-MXFP4-loom

Qwen3.8-27B from the published MXFP4 GGUF (quark75/Qwen3.8-27B-MXFP4-GGUF: MXFP4 on every dense linear of the GDN layers and the FFN, Q8_0 on the
16 attention layers' projections, the head, the embeddings and the small SSM tensors, the MTP head kept; 16.9 GB), a second package of the model beside the
Q4_K_M card. Vision comes from the bartowski mmproj (the projector does not depend on the text quant).

Radeon AI PRO R9700 serving settings (prefill_capacity 1024) are those of the Q4_K_M package; resolved per device by `resolve.py`, written into the
`<first>-r9700` preset of model.json by `make_info.py build --fit`.

Settings this node turns on (resolved by `scope.json`): `prefill.gemm.mxfp4_f8` on gfx12 (the R9700's prefill GEMMs of the MXFP4 tensors run on fp8 WMMA; the gfx11 programs
keep the f16 GEMMs) and `decode.mxfp4.act_q8` (int8-activation dot4 matvecs: the f32 MXFP4 matvec made the 4-token verify of a drafted decode ALU-bound, 47 tok/s
instead of 73.8 on the R9700, run 43761307081). Both are opt-in per model: gpt-oss's MoE experts keep their f32 MXFP4 kernels.
