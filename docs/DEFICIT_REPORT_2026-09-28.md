# Photorealism deficit report — 2026-09-28

Instrument: `scripts/harness/run_gates.py` against the frozen
`research/avenues/gates_calibration.json` (n=1000 real FFHQ,
seed 42). Bands are [p5, p95] of the real reference.

| gate | band [p5, p95] | real (A) | **VAE ceiling (C)** | EIR recon (B) | EIR sweep | arm-j | dip-conv-head |
|---|---|---|---|---|---|---|---|
| `g0a_sensor_noise_floor` | [-1.7292, -1.3546] | -1.5544 ✅ | -1.5727 ✅ | — | — | — | — |
| `g0a_sensor_noise_floor_full` | [-1.5360, -1.2225] | -1.3766 ✅ | -1.3935 ✅ | -1.3082 ✅ | -1.2672 ✅ | -1.5529 ❌ | -1.6929 ❌ |
| `g0b_spectral_slope` | [-1.6085, -1.0926] | -1.3572 ✅ | -1.3778 ✅ | -1.2057 ✅ | -1.3032 ✅ | -1.4660 ✅ | -1.3240 ✅ |
| `g0c_skin_texture_energy` | [0.0010, 0.0084] | 0.0034 ✅ | 0.0032 ✅ | — | — | — | — |
| `g0c_skin_texture_energy_full` | [0.0029, 0.0165] | 0.0084 ✅ | 0.0078 ✅ | 0.0107 ✅ | 0.0109 ✅ | 0.0025 ❌ | 0.0142 ✅ |
| `g0d_local_contrast_full_p5` | [0.0024, 0.0280] | 0.0138 ✅ | 0.0135 ✅ | 0.0325 ❌ | 0.0138 ✅ | 0.0109 ✅ | 0.0098 ✅ |
| `g0d_local_contrast_full_p50` | [0.0460, 0.1136] | 0.0775 ✅ | 0.0756 ✅ | 0.0878 ✅ | 0.0927 ✅ | 0.0521 ✅ | 0.0318 ❌ |
| `g0d_local_contrast_full_p95` | [0.1562, 0.3095] | 0.2223 ✅ | 0.2170 ✅ | 0.1773 ✅ | 0.1954 ✅ | 0.1915 ✅ | 0.1199 ❌ |
| `g0d_local_contrast_p5` | [0.0030, 0.0311] | 0.0169 ✅ | 0.0165 ✅ | — | — | — | — |
| `g0d_local_contrast_p50` | [0.0474, 0.1208] | 0.0815 ✅ | 0.0795 ✅ | — | — | — | — |
| `g0d_local_contrast_p95` | [0.1518, 0.3115] | 0.2210 ✅ | 0.2158 ✅ | — | — | — | — |

## Counts in band

| leg | in band | out of band | of total |
|---|---|---|---|
| real (A) | 11 | 0 | 11 |
| vae-ceiling (C) | 11 | 0 | 11 |
| eir-recon (B) | 5 | 1 | 6 |
| eir-sweep | 6 | 0 | 6 |
| arm-j | 4 | 2 | 6 |
| dip-conv-head | 3 | 3 | 6 |
| gamma2-noise-scale | 2 | 4 | 6 |
