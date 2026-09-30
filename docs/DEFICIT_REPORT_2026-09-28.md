# Photorealism deficit report — 2026-09-28

Instrument: `scripts/harness/run_gates.py` against the frozen
`research/avenues/gates_calibration.json` (n=1000 real FFHQ,
seed 42). Bands are [p5, p95] of the real reference.

| gate | band [p5, p95] | real (A) | **VAE ceiling (C)** | EIR recon (B) | EIR sweep | arm-j | dip-conv-head |
|---|---|---|---|---|---|---|---|
| `g0a_sensor_noise_floor` | [-1.7292, -1.3546] | -1.5544 ✅ | -1.5727 ✅ | -1.5273 ✅ *(n=126)* | -1.6788 ✅ *(n=18)* | — | — |
| `g0a_sensor_noise_floor_full` | [-1.5360, -1.2225] | -1.3766 ✅ | -1.3935 ✅ | -1.3082 ✅ | -1.2672 ✅ *(n=18)* | -1.5529 ❌ *(n=10)* | -1.6929 ❌ *(n=11)* |
| `g0b_spectral_slope` | [-1.6085, -1.0926] | -1.3572 ✅ | -1.3778 ✅ | -1.2057 ✅ | -1.3032 ✅ *(n=18)* | -1.4660 ✅ *(n=10)* | -1.3240 ✅ *(n=11)* |
| `g0c_skin_texture_energy` | [0.0010, 0.0084] | 0.0034 ✅ | 0.0032 ✅ | 0.0038 ✅ *(n=126)* | 0.0015 ✅ *(n=18)* | — | — |
| `g0c_skin_texture_energy_full` | [0.0029, 0.0165] | 0.0084 ✅ | 0.0078 ✅ | 0.0107 ✅ | 0.0109 ✅ *(n=18)* | 0.0025 ❌ *(n=10)* | 0.0142 ✅ *(n=11)* |
| `g0d_local_contrast_full_p5` | [0.0024, 0.0280] | 0.0138 ✅ | 0.0135 ✅ | 0.0325 ❌ | 0.0138 ✅ *(n=18)* | 0.0109 ✅ *(n=10)* | 0.0098 ✅ *(n=11)* |
| `g0d_local_contrast_full_p50` | [0.0460, 0.1136] | 0.0775 ✅ | 0.0756 ✅ | 0.0878 ✅ | 0.0927 ✅ *(n=18)* | 0.0521 ✅ *(n=10)* | 0.0318 ❌ *(n=11)* |
| `g0d_local_contrast_full_p95` | [0.1562, 0.3095] | 0.2223 ✅ | 0.2170 ✅ | 0.1773 ✅ | 0.1954 ✅ *(n=18)* | 0.1915 ✅ *(n=10)* | 0.1199 ❌ *(n=11)* |
| `g0d_local_contrast_p5` | [0.0030, 0.0311] | 0.0169 ✅ | 0.0165 ✅ | 0.0308 ✅ *(n=122)* | 0.0123 ✅ *(n=18)* | — | — |
| `g0d_local_contrast_p50` | [0.0474, 0.1208] | 0.0815 ✅ | 0.0795 ✅ | 0.0854 ✅ *(n=122)* | 0.0897 ✅ *(n=18)* | — | — |
| `g0d_local_contrast_p95` | [0.1518, 0.3115] | 0.2210 ✅ | 0.2158 ✅ | 0.1749 ✅ *(n=122)* | 0.1983 ✅ *(n=18)* | — | — |

## Counts in band

| leg | in band | out of band | of total |
|---|---|---|---|
| real (A) | 11 | 0 | 11 |
| vae-ceiling (C) | 11 | 0 | 11 |
| eir-recon (B) | 10 | 1 | 11 |
| eir-sweep | 11 | 0 | 11 |
| arm-j | 4 | 2 | 6 |
| dip-conv-head | 3 | 3 | 6 |
| gamma2-noise-scale | 2 | 4 | 6 |

## Out-of-band gates, with their own noise

`z` = gap to the nearest band edge / (std/sqrt(n)). |z| < 2 means the number is not
separable from the band edge on its own sample — an out-of-band label, not evidence.

| leg | gate | value | nearest edge | gap | n | z | reading |
|---|---|---|---|---|---|---|---|
| eir-recon (B) | `g0d_local_contrast_full_p5` | 0.0325 | 0.0280 | 0.0044 | 160 | 4.79 | beyond its own noise |
| arm-j | `g0a_sensor_noise_floor_full` | -1.5529 | -1.5360 | 0.0168 | 10 | 0.68 | not separable from the edge |
| arm-j | `g0c_skin_texture_energy_full` | 0.0025 | 0.0029 | 0.0003 | 10 | 1.23 | not separable from the edge |
| dip-conv-head | `g0a_sensor_noise_floor_full` | -1.6929 | -1.5360 | 0.1568 | 11 | 5.87 | beyond its own noise |
| dip-conv-head | `g0d_local_contrast_full_p50` | 0.0318 | 0.0460 | 0.0142 | 11 | 13.15 | beyond its own noise |
| dip-conv-head | `g0d_local_contrast_full_p95` | 0.1199 | 0.1562 | 0.0363 | 11 | 4.11 | beyond its own noise |
| gamma2-noise-scale | `g0a_sensor_noise_floor_full` | -1.7707 | -1.5360 | 0.2347 | 11 | 7.93 | beyond its own noise |
| gamma2-noise-scale | `g0b_spectral_slope` | -1.8071 | -1.6085 | 0.1986 | 11 | 5.79 | beyond its own noise |
| gamma2-noise-scale | `g0d_local_contrast_full_p50` | 0.0316 | 0.0460 | 0.0144 | 11 | 9.77 | beyond its own noise |
| gamma2-noise-scale | `g0d_local_contrast_full_p95` | 0.1021 | 0.1562 | 0.0541 | 11 | 7.14 | beyond its own noise |

## Detection attrition (the defect that hides in the denominator)

| leg | panels on disk | images with a usable skin mask | attrition |
|---|---|---|---|
| real (A) | 150 | 150 | 0.0% |
| vae-ceiling (C) | 150 | 150 | 0.0% |
| eir-recon (B) | 160 | 126 | 21.2% |
| eir-sweep | 18 | 18 | 0.0% |

`eir-recon` attrition is concentrated at step 1000 (11 of 16 panels) but persists
at 1-4 panels per later boundary; real faces show none at all.
