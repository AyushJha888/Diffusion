import argparse
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MotionSequenceDataset
from diffusion import GaussianDiffusion
from models import PositionalUNet1D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a motion diffusion model on a single motion clip.")
    parser.add_argument("--json-path", type=str, default="data/uptown_funk.json")
    parser.add_argument("--window", type=int, default=160, help="Sequence length for training windows.")
    parser.add_argument("--stride", type=int, default=20, help="Stride between windows when building the dataset.")
    parser.add_argument("--limit", type=int, default=512, help="Optional max number of windows to sample.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=400, help="Number of diffusion steps.")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Override device selection (cpu/cuda).")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def prepare_dataloader(args: argparse.Namespace) -> MotionSequenceDataset:
    dataset = MotionSequenceDataset(
        json_path=args.json_path,
        window=args.window,
        stride=args.stride,
        limit=args.limit,
        seed=args.seed,
    )
    return dataset


def save_checkpoint(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: MotionSequenceDataset,
    optimizer: torch.optim.Optimizer,
    save_path: Path,
    step: int,
    extra_config: Dict[str, Any],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "diffusion_config": {
            "timesteps": diffusion.timesteps,
            "beta_start": diffusion.betas[0].item(),
            "beta_end": diffusion.betas[-1].item(),
        },
        "data_meta": {
            "mean": dataset.mean.cpu(),
            "std": dataset.std.cpu(),
            "key_order": dataset.key_order,
            "sequence_length": dataset.sequence_length,
            "num_channels": dataset.num_channels,
        },
        "step": step,
        "extra_config": extra_config,
    }
    torch.save(payload, save_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = prepare_dataloader(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PositionalUNet1D(in_channels=dataset.num_channels)
    diffusion = GaussianDiffusion(model, timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in dataloader:
            # Ensure batch is shaped (B, C, T) before diffusion
            if batch.dim() == 2:
                batch = batch.unsqueeze(0)
            elif batch.dim() == 4 and batch.shape[1] == 1:
                batch = batch.squeeze(1)
            if batch.dim() != 3:
                raise ValueError(f"Expected batch shape (B, C, T), got {batch.shape}")

            batch = batch.to(device)
            t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=device)
            loss = diffusion.p_losses(batch, t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if global_step % args.log_interval == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
            global_step += 1

        ckpt_path = Path(args.save_dir) / f"ddpm_epoch_{epoch+1}.pth"
        save_checkpoint(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            optimizer=optimizer,
            save_path=ckpt_path,
            step=global_step,
            extra_config=vars(args),
        )
        print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
