# How to Use This Repository

The project trains and samples a diffusion model for 2D motion keypoints extracted from a single clip (`data/uptown_funk.json`). Follow the steps below to reproduce the experiments or run your own.

## 1. Clone & install
```bash
git clone <your-fork-or-this-repo>
cd Diffusion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare the dataset
The repo already ships with `data/uptown_funk.json`. If you have another motion clip in the same JSON format (dict of joint name -> list of `[x, y]`), drop it in `data/` and point `--json-path` to it.

## 3. Train the diffusion model
Use `train.py` and customize the window/stride/limit arguments to control how many overlapping windows you slice from the full sequence.

```bash
python train.py \
  --json-path data/uptown_funk.json \
  --window 256 \
  --stride 5 \
  --limit 1024 \
  --batch-size 16 \
  --epochs 1000 \
  --lr 2e-4 \
  --timesteps 400 \
  --beta-start 1e-4 \
  --beta-end 0.02 \
  --save-dir checkpoints/1000 \
  --num-workers 4 \
  --seed 0 \
  --patience 0 \
  --loss-curve-path logs/loss_curve.png
```

Key flags:
- `--window`, `--stride`, `--limit`: define how many clips (`N`) you sample from the original sequence and how much they overlap.
- `--patience 0`: disable early stopping; set to a positive value to enable it.
- `--loss-curve-path`: optional path to save a loss curve PNG.

Checkpoints (and the normalization stats they contain) land under `--save-dir`.

## 4. Sample new motion clips
Invoke `sample.py` with any saved checkpoint to generate JSON clips, optionally rendering GIFs via `renderer.py`.

```bash
python sample.py \
  --checkpoint checkpoints/1000/ddpm_best.pth \
  --num-samples 4 \
  --out-dir samples/1000 \
  --render-gif \
  --fps 10
```

Outputs:
- `sample_{i}.json`: unnormalized keypoints ready for downstream use.
- `sample_{i}.gif`: visualizations (requires `--render-gif`).


