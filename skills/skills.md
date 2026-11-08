## Confirmed (passed held-out)
- [0001] Raising AdamW learning rate for faster convergence within fixed time budget succeeded (dev -2.8 mm).
- [0002] Sample t from logit-normal to focus on hard intermediate timesteps; dev error -2.7 mm — use for velocity training.
- [0014] Lesson: Use an EMA of weights at sampling time for the flow model—smoother, better-generalizing output at no extra parameter cost; accepted, dev improved by -3.4 mm.
- [0017] Lesson: FiLM modulation of each residual block by the timestep and 2D condition beat input-only conditioning, cutting dev error by 1.5 mm—adopt per-block modulation.
## Failed ideas (do not retry without a new reason)
- [0003] Increasing to width 640 with 5 blocks to stay under 5M params gave +1.8mm dev error with no gain — rejected; blind capacity adds overfit risk.
- [0004] Lesson: Zeroing invisible 2D joints to block occluded noise while keeping mask signal gave no gain (+0.8mm dev) — rejected.
- [0005] Visibility-weighted velocity loss gave no gain (-0.2mm): occluded joints add little noise, visible joints already dominate.
- [0006] Lesson: AdamW weight decay to curb pose overfitting gave no gain, dev +2.6 mm — rejected; avoid weight decay for this.
- [0008] Raising AdamW to 1e-3 for this LayerNorm MLP gave only +1.9 mm dev gain in the fixed budget (rejected: no_gain), yielding no useful faster convergence.
- [0009] Narrowing the logit-normal to concentrate on hard mid-range timesteps for velocity prediction gave no gain (+1.9 mm worse), so keep broader timestep sampling.
- [0010] Lesson: Per-joint tokens with self-attention for skeleton structure/joint dependencies gave no gain vs a flat pose vector, worsening dev error by +12.0 mm; avoid this architecture.
- [0011] Centering/scaling 2D keypoints by mask-weighted centroid/spread (keeping position via frame features) gave -0.5 mm on Dev: rejected; internal normalization adds no conditioning gain.
- [0012] Replacing velocity MSE with Euclidean loss on the implied clean pose gave +0.0 mm: directly matching the eval metric doesn't beat velocity-space training.
- [0013] Dropout inside each residual block didn't reduce pose overfitting (dev +3.6 mm, rejected); no gain despite unchanged interface/params—avoid intra-block dropout for this regularization.
- [0015] Lesson: tune_optimizer at LR 6e-4 (between 3e-4 and unstable 1e-3) gave no gain, -0.5 mm dev, rejected—don't retry intermediate LR under same budget.
- [0016] Lesson: Broadening logit-normal timestep sampling beyond sigma=1 (opposite of failed narrowing) gave no gain (+1.7 mm dev, rejected) — don't widen endpoint coverage.
- [0018] Adding explicit 2D bone-vector differences to conditioner gave no gain (-0.2 mm dev): raw joint coordinates already encode limb direction/length for the MLP.
- [0019] Huber velocity loss gave no gain (-0.9 mm, rejected): down-weighting rare extreme pose/occlusion outliers doesn't help; MSE's smooth gradients for typical errors suffice.
- [0020] Lesson: Adding small Gaussian noise only to visible 2D keypoints gave no gain (+3.2 mm dev worsening), so it doesn't regularize against detector localization error on unseen data.
- [0022] Lesson: Retrying the unevaluated cosine schedule (tune_optimizer) gave no gain—dev worsened +2.1 mm and was rejected; it doesn't improve the final minimum under a fixed budget.
- [0023] Uniform timestep sampling vs logit-normal: rejected (+2.9 mm worse); even endpoint coverage didn't improve ODE integration—keep mid-range timestep bias.
- [0024] Gated linear units in residual blocks for multiplicative feature interactions gave no gain (-1.5 mm dev, rejected); plain GELU MLP suffices within the parameter limit.
## Open hypotheses
- [hint] The time budget is fixed, so more training steps per second lowers error: try bf16 autocast, a larger batch with a matching learning rate, or sampling batch indices on the GPU instead of with numpy.
- [hint] 0005 (weight the velocity loss by 2D visibility) scored 72.29, just short of the bar; try combining it with another change.
