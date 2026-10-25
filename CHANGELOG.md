# Changelog

## v0.2.0 — 2026-10-25

- Add full-resolution 28×28 image-conditioned U-Net and matched-data MLP controls.
- Train two U-Net seeds, two MLP256 seeds and one wider MLP1024 control for 3,000 updates each.
- Support editable observed pixels, binary masks, exact observation projection, image upload and noise editing.
- Publish generation, diversity, local CNN feature metrics, held-out half-image completion and paste baselines.
- Add reproducible checkpoint recovery, actual HTTP inference checks and CPU CI.
- Preserve the original three-seed 14×14 experiments, generated arrays and research report.
- Distribute verified models and raw arrays as release assets.

Browser pointer/upload automation remains unverified: Chromium exited during page navigation in the experiment environment. CPU inference and HTTP checks completed.
