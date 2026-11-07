## Confirmed (passed held-out)
- [0001] Raising AdamW learning rate for faster convergence within fixed time budget succeeded (dev -2.8 mm).
- [0002] Sample t from logit-normal to focus on hard intermediate timesteps; dev error -2.7 mm — use for velocity training.
## Failed ideas (do not retry without a new reason)
- [0003] Increasing to width 640 with 5 blocks to stay under 5M params gave +1.8mm dev error with no gain — rejected; blind capacity adds overfit risk.
- [0004] Lesson: Zeroing invisible 2D joints to block occluded noise while keeping mask signal gave no gain (+0.8mm dev) — rejected.
- [0005] Visibility-weighted velocity loss gave no gain (-0.2mm): occluded joints add little noise, visible joints already dominate.
## Open hypotheses
