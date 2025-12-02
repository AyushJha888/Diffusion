import argparse
import os
import random
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("--loss-curve-path", type=str, default=None, help="Where to save a loss curve image.")
    parser.add_argument("--device", type=str, default=None, help="Override device selection (cpu/cuda).")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without improvement). Default: 20. Set to 0 to disable.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum change to qualify as an improvement.")
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
    epoch: int,
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
        "epoch": epoch,
        "extra_config": extra_config,
    }
    torch.save(payload, save_path)


def plot_loss_curve(loss_values: List[float], save_path: Path) -> None:
    if not loss_values:
        return
    save_path = save_path if isinstance(save_path, Path) else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(loss_values)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
    diffusion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    global_step = 0
    loss_history: List[float] = []
    
    # Early stopping setup (enabled by default)
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = args.patience if args.patience > 0 else float('inf')
    min_delta = args.min_delta
    recent_losses = []  # Track recent losses for plateau detection
    window_size = min(5, max(3, patience // 2)) if patience < float('inf') else 0  # Look at last few epochs for stability
    
    if patience < float('inf'):
        print(f"Early stopping enabled: will stop after {patience} epochs without improvement (min_delta={min_delta})")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
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

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            loss_history.append(loss_value)
            if global_step % args.log_interval == 0:
                print(f"epoch {epoch} step {global_step} loss {loss_value:.4f}")
            global_step += 1
        
        # Calculate average loss for the epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"epoch {epoch} completed - average loss: {avg_epoch_loss:.4f}")
        
        # Track recent losses for plateau detection (only if early stopping is enabled)
        if patience < float('inf'):
            recent_losses.append(avg_epoch_loss)
            if len(recent_losses) > window_size:
                recent_losses.pop(0)
        
        # Early stopping check
        improved = avg_epoch_loss < best_loss - min_delta
        if improved:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model checkpoint
            ckpt_path = Path(args.save_dir) / "ddpm_best.pth"
            save_checkpoint(
                model=model,
                diffusion=diffusion,
                dataset=dataset,
                optimizer=optimizer,
                save_path=ckpt_path,
                step=global_step,
                epoch=epoch,
                extra_config=vars(args),
            )
            print(f"saved best checkpoint (loss: {best_loss:.4f}) to {ckpt_path}")
        else:
            patience_counter += 1
            
            # Check for plateau: if recent losses are stable (low variance), stop earlier
            plateau_detected = False
            if patience < float('inf') and len(recent_losses) >= window_size and patience_counter >= window_size:
                loss_variance = np.var(recent_losses)
                loss_range = max(recent_losses) - min(recent_losses)
                # If losses are very stable (low variance and small range), consider it a plateau
                if loss_variance < min_delta * 2 and loss_range < min_delta * 3:
                    plateau_detected = True
                    print(f"  Plateau detected: loss has stabilized (variance={loss_variance:.6f}, range={loss_range:.6f})")
            
            if patience_counter >= patience or plateau_detected:
                reason = "plateau detected" if plateau_detected else f"{patience} epochs without improvement"
                print(f"\nEarly stopping triggered: {reason}.")
                print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
                print(f"Stopping training at epoch {epoch + 1}")
                break
            elif patience < float('inf'):
                print(f"  (no improvement for {patience_counter}/{patience} epochs, best: {best_loss:.4f})")

        # Save checkpoint every 50 epochs or on the last epoch
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            ckpt_path = Path(args.save_dir) / f"ddpm_epoch_{epoch+1}.pth"
            save_checkpoint(
                model=model,
                diffusion=diffusion,
                dataset=dataset,
                optimizer=optimizer,
                save_path=ckpt_path,
                step=global_step,
                epoch=epoch,
                extra_config=vars(args),
            )
            print(f"saved checkpoint to {ckpt_path}")

    if args.loss_curve_path:
        plot_loss_curve(loss_history, Path(args.loss_curve_path))


if __name__ == "__main__":
    main()
