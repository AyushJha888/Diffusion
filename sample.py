import argparse
import json
from pathlib import Path
from typing import List

import torch

from dataset import sequence_to_keypoint_dict
from diffusion import GaussianDiffusion
from models import PositionalUNet1D
import renderer


def load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    required = ["model_state", "diffusion_config", "data_meta"]
    for key in required:
        if key not in ckpt:
            raise ValueError(f"checkpoint missing key '{key}'")
    return ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample sequences from a trained motion diffusion model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="samples")
    parser.add_argument("--render-gif", action="store_true", help="Render .gif files using renderer.py.")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, device)

    data_meta = ckpt["data_meta"]
    mean = data_meta["mean"].to(device)
    std = data_meta["std"].to(device)
    key_order: List[str] = data_meta["key_order"]
    seq_len = int(data_meta["sequence_length"])
    num_channels = int(data_meta["num_channels"])

    model = PositionalUNet1D(in_channels=num_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    diff_cfg = ckpt["diffusion_config"]
    diffusion = GaussianDiffusion(
        model,
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
    )
    diffusion.to(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        samples = diffusion.sample(
            shape=(args.num_samples, num_channels, seq_len),
            device=device,
        )

    for idx, seq in enumerate(samples):
        kp_dict = sequence_to_keypoint_dict(seq, key_order=key_order, mean=mean, std=std)
        json_path = out_dir / f"sample_{idx}.json"
        with json_path.open("w") as f:
            json.dump(kp_dict, f)
        print(f"wrote {json_path}")

        if args.render_gif:
            gif_path = out_dir / f"sample_{idx}.gif"
            renderer.render_seq(kp_dict, str(gif_path))
            print(f"rendered gif to {gif_path}")


if __name__ == "__main__":
    main()
