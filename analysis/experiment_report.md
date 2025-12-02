# Motion Diffusion Experiments

## Setup
- **Data.** All runs use the single-clip `data/uptown_funk.json` keypoint dataset, processed through `MotionSequenceDataset` (`dataset.py`) to build fixed-length windows and normalize them before training.
- **Model + diffusion process.** The noise-prediction network is `PositionalUNet1D` with sinusoidal timestep embeddings and a light 1D UNet topology (`models.py`). Training and sampling follow the Gaussian DDPM wrapper in `diffusion.py` with 400 diffusion steps and linear beta schedule.
- **Optimization.** `train.py` trains with Adam (lr `2e-4`), grad clipping at `1.0`, and optional early-stopping. Checkpoints store normalization stats so we can sample and unnormalize later.

## Evaluation protocol
To compare runs, I sampled 8 motion clips from each checkpoint on CPU (seed 42) using the repository's diffusion/sampling code. Each sample is unnormalized with the checkpoint's mean/std. I then computed simple motion descriptors:

- `coord_range`: mean global range (max − min) per coordinate channel.
- `pos_std`: average standard deviation per coordinate channel.
- `vel_std` / `vel_energy`: standard deviation and mean-squared magnitude of frame-to-frame velocities.
- `com_speed`: mean speed of the skeleton's center of mass.
- `diversity`: average pairwise L2 distance between flattened sequences, normalized by dimensionality.

For reference I measured the same statistics on the training data windows (unnormalized) that correspond to each configuration.

## Dataset baselines
| split | window/stride/limit | coord_range | pos_std | vel_std | vel_energy | com_speed | diversity | clips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset_64x32x32 | 64 / 32 / 32 | 245.10 | 40.87 | 7.32 | 64.28 | 5.31 | 60.17 | 26 |
| dataset_256x5x1024 | 256 / 5 / 1024 | 245.10 | 41.74 | 7.32 | 65.03 | 5.50 | 57.95 | 128 |

## Model checkpoints evaluated
| tag | checkpoint | training config (window / stride / limit / epochs) | coord_range | pos_std | vel_std | vel_energy | com_speed | diversity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_1epoch | `checkpoints/ddpm_epoch_1.pth` | 64 / 32 / 32 / 1 | **2599.05** | 436.14 | 617.15 | 424483.91 | 188.58 | 650.00 |
| earlystop_500 | `checkpoints/ddpm_epoch_500.pth` | 256 / 5 / 1024 / 500 | 181.80 | 22.39 | 5.30 | 33.58 | 2.96 | 34.78 |
| long_1000 | `checkpoints/1000/ddpm_best.pth` | 256 / 5 / 1024 / 1000 | 182.34 | 26.02 | 5.84 | 41.21 | 3.45 | 39.11 |
| long_10000 | `checkpoints/10000/ddpm_best.pth` | 256 / 5 / 1024 / 10000 | 192.45 | 28.22 | 5.24 | 34.30 | 3.58 | 42.36 |

## Observations
1. **Sanity check (1-epoch run).** Training on only 26 short clips for a single epoch produces wildly unbounded outputs (ranges >10× the ground truth). The diffusion model has not learned the dataset statistics at all; even the center-of-mass speed is two orders of magnitude too large. This run is only useful for pipeline debugging.
2. **500-epoch run (no early stopping).** Moving to the 256-frame dataset with much denser windows immediately brings all statistics into the right ballpark. However, every metric is still below the real data: the model underestimates positional variance (22 vs 41) and produces motion that is ~50% slower (COM speed 2.96 vs 5.50). Diversity across samples also drops to ~60% of the dataset's value, signaling mode collapse toward a limited subset of moves.
3. **1000-epoch run.** Training longer continues to close the gap. Positional variance rises to 26, velocity variance increases, and sample diversity climbs to 39. Nevertheless, the COM speed and velocity magnitude remain low (≈60% of reference), so gestures still look damped compared to the dance clip.
4. **10000-epoch run.** Extremely long training yields diminishing returns. Coord range and positional std tick up slightly, but velocity-related metrics fluctuate (vel_std decreases, com_speed barely improves). Diversity reaches 42—better but still well under the dataset baseline. The plateau likely stems from training on a single clip: after a point the model memorizes the limited motion statistics available.

## Takeaways and next steps
- **Data coverage is the main bottleneck.** All long runs plateau below the dataset's own variance/diversity values, so adding more motion clips or more aggressive window jittering would likely help more than pushing epochs higher.
- **Augment velocity fidelity.** The gap in `vel_std` and `com_speed` suggests the diffusion loss is biased toward static poses. Adding a velocity-weighted term (e.g., predict first-order differences jointly) or training on shorter windows with overlap could encourage sharper dynamics.
- **Monitoring.** Capturing per-epoch metrics during training (e.g., by logging loss/variance stats to TensorBoard instead of just saving checkpoints) would make it easier to spot the plateau much earlier than 10k epochs.
