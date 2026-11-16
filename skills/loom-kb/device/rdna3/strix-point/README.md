# Strix Point (gfx1150, Radeon 890M)

Ryzen AI 300 series, 16 RDNA 3.5 CUs, wave32 WMMA like Strix Halo, **no MALL** (GPU L2 only). GPU memory is system RAM through GTT.
Measured on one laptop: HP OmniBook Ultra 14 (Ryzen AI 9 HX 375, 32 GB LPDDR5X-7500, 128-bit: 120 GB/s peak), Ubuntu kernel 7.0,
hrx-loom 0.1.0 from PyPI, 2026-11-16. No DevLab runner: numbers come from `.github/scripts/devlab_perf.sh` run locally.
Per-model results (fit, budget, rows, settings): `fleet/devices/gfx1150.json`.

- Family `rdna3` (gfx11-generic): the published programs compile and run unchanged (Qwen3.5-4B card: registration 3 s, func, dev gates all
  PASS). No new kernels.
- `kb.py route --device gfx1150` (or `strix-point`, `890m`) routes here; roofline preset `strix-point` (`roofline.py --device strix-point`).

## Host setup it needs (model-independent, not in the card)

1. **Locked-memory limit.** loom-serve streams weights through io_uring registered buffers; Ubuntu's default `RLIMIT_MEMLOCK` (8 MiB, also the
   hard limit) makes every weight load fail AFTER registration reports `ready`: the server log fills with
   `driver_failure ... io_uring/proactor_registration.c:358: RESOURCE_EXHAUSTED; buffer pin preparation failed (12)` and every request
   returns `model is unavailable`. Fix (root, then log in again): `/etc/security/limits.d/90-loom.conf` with `* - memlock unlimited`
   (and `DefaultLimitMEMLOCK=infinity` in `/etc/systemd/{system,user}.conf` for services). Without root, a preload that makes
   `io_uring_setup` return ENOSYS sends IREE to its portable threaded backend and serving works (how the numbers below were taken);
   it is a diagnostic, not a deployment. Check: `ulimit -l` must print `unlimited` (or several GiB).
2. **Memory budget = GTT, not 60 % of RAM.** GTT defaults to half of RAM (15.08 GiB on 32 GB). A card's Strix Halo quick start
   (`--memory_bytes=28000000000`) registers and then fails every allocation; `--pool_backing=fixed` fails at registration (a 262k-context
   preset commits its whole pool: `libamdf ERRNO 0x0000000C`). Serve elastic with `--memory_bytes` <= GTT (14-15e9 on 32 GB). Too small
   (12e9 for Qwen3.5-4B) fails registration with code 8. Read the budget from `/sys/class/drm/card*/device/mem_info_gtt_total`.

## Measured (Qwen3.5-4B Q4_K_M, published card unmodified, DFlash depth 3, `--memory_bytes=15000000000`)

| row | Loom | llama.cpp ROCm b1374 | llama.cpp Vulkan b12211 | Strix Halo card (Loom) |
|---|---|---|---|---|
| decode | 31.3 | 20.6 | 20.6 | 87.4 |
| decode @2k | 29.1 | 25.0 | 22.9 | 84.1 |
| decode @32k | 22.2 | 15.8 | 14.5 | 60.9 |
| prefill 1k | 997 | 698 | 590 | 2840 |
| prefill 2k | 1010 | 697 | 648 | 2906 |
| 4 streams (no drafting) | 81.6 | 73.9 | 68.1 | 223.4 |

Drafted acceptance 0.33 / 0.42 / 0.40 (decode / 2k / 32k), like Strix Halo's. Loom/llama.cpp margins are the Halo card's shape
(+52 % decode, +43 % prefill, +10 % c4); absolute rates are 0.36x Halo for decode and prefill: below the 0.56x bandwidth ratio (120/216),
close to the 0.40x compute ratio. Not yet measured: prefill 8k/16k/32k, a measured streaming bandwidth, and a per-device preset sweep.
