# Facet: hybrid-gdn: decode

## Decode vs prefill split of the recurrence kernels

  - Decode vs prefill split (MiniCPM-V-4.7 / Qwen3.5-35B-A3B, DevLab, active wait): spans shorter than 64 tokens
    (decode, verify) run `g_gdn_recurrence` with RL 16 x 2 columns, `GDN_CONV_UNROLL=8`; longer spans run
    `g_gdn_recurrence_pf` (RL 8 x 4 columns, the 2k winner above). Uniform 16 x 2 gave decode +5-8 % but prefill -12 to
    -16 % on the MoE hybrids (-3 % on Qwen3.5-9B), hence two kernels; a span runs in exactly one of them, chosen by its own length, so batching never changes its numerics. `GDN_ROW_LANES`/`GDN_COLS` set in the environment
    force one form for both.

## Fusing the GDN front of a drafted step does not pay (R9700, gfx12, Qwen3.8-27B, q38r9700)

  - Doubling (run 43770401446, MTP 3 step 35.4 ms): conv_commit 1.08 ms, conv + commit + qk_norm 1.60 ms, recurrence 0.05 (verify) /
    0.03 (replay), norm_gate 0.14 per step. The capture / restore copies were the cheap win (#866, `decode.spec.gdn_copy_split`, +2 %).
  - A fused gfx12 front (conv + history commit + q/k norms in one launch; ktest, gates and text match pass) LOST 4 % decode (v1 128-thread
    blocks, run 43771517926; v2 one wave per 128 channels, run 43774217524): the fused kernel costs ~18 us per launch, the three legacy
    kernels ~16.6 us together. The chain is latency-bound (dependent metadata / origin / history loads), not launch-bound. Not merged.
