# Model: bartowski-Qwen3.8-27B-loom

Radeon AI PRO R9700 serving settings of `bartowski-Qwen3.8-27B-loom` (prefill_capacity 1024); resolved per device by `resolve.py`, written into the `<first>-r9700` preset of model.json by `make_info.py build --fit`.

MTP proposals (depth 3) read a Q4_0 copy of the head's first 81920 rows plus the added-token rows (`draft.propose.head_rows`, `draft/prep_weights.py`; load-time conversion of the published Q6_K head, verification keeps the Q6_K head), 0.2 GB per proposal round instead of 1.04 GB. Same package on both GPUs.

## Limits of hrx-loom 0.1.0 met on the R9700 (q38r9700 campaign, 2026-11-15)
- **Verify width.** `draft_depth` is 1-7 (`topic/speculative.md`), so a drafted step verifies at most 8 rows. The z-lab DFlash2 drafter
  widened from block 8 to 16 (Lucebox on the R9700: HumanEval 134 -> 208 tok/s, math +17-22 %, prose unchanged) cannot be expressed;
  measured at depth <= 7 DFlash2 beat MTP 3 only on short decode (+13 %), tied at 2k, lost at 32k and costs context (below), so the card
  drafts with MTP. Unlock: depths up to 15 (16-row verify).
- **Context with a drafter.** With `--pool_backing=fixed` (required on the R9700, see `device/rdna4/radeon-r9700`) the pool is committed
  eagerly: the DFlash2 drafter drops the served context 124928 -> 83968 at depth 3 and 40960 at depth 5/7 (RESOURCE_EXHAUSTED); the MTP
  proposal-head copy costs 124928 -> 120832 (image component unavailable above it, dev gates). Not an SDK bug; it bounds drafter choice on 32 GB.
- Not a limit: fp8 WMMA (`v_wmma_f32_16x16x16_fp8_fp8`, `_bf8_fp8`) compiles for gfx1201 with 0.1.0 (used by the MXFP4 card, never by Q4_K_M weights).
