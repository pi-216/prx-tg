# Red Lines — prx-tg (Founder-Owned)

This file is owned exclusively by the founder (Tim). Agents may read it, never edit it.
Any suggestion or experiment arm that violates a rule below is ruled out forever.

1. **No warm-starting from tainted P1 (`latent-first-pretrain`) or PP (`pixel-posttrain`) checkpoints.** Both arms memorized DINOv3 CLS tokens as per-image lookup keys on unsplit data while ignoring text conditioning.
2. **No SD 1.5 geometry-adapter arms (`geometry-adapter-sd15`, `geo-adapter-*`).** Discontinued permanently (`docs/DISCONTINUATION_NOTICE_SD15.md`) after 40k+ cumulative steps proved the frozen CLIP/UNet path ignores geometry tokens.
3. **No real-person likeness or recognizer R@1 against real people as a product goal.** Identity in `prx-tg` means **consistency across renders** of a point in the identity space across seeds and poses, never reproducing a real person's photograph (`docs/EXPERIMENT_TREE.md` Goal root node).
4. **No training runs without a pre-registered structural/anatomy gate and person-held-out split.** Pairwise Elo and G0 spectral bands alone are proven blind to absent facial anatomy (`experiment-configs/eir-cosine-restart/audit/15k_review/RESULTS.md`).
5. **No soft governance files (`config.yaml`, `provenance.yaml`, `README.md`) stored only under the `experiments/` NAS symlink.** Every arm's soft trio must live in git under `experiment-configs/<slug>/`.
6. **No bespoke systemd timers (`*-launcher.timer`, `*-heartbeat.timer`).** All work is scheduled via `ds4-queue` and bounded GPU runs via `bin/gpu_run` (`gpu-scheduler`).
