import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _dict_to_array(data: Dict[str, Sequence[Sequence[float]]], key_order: List[str]) -> np.ndarray:
    """Convert a {key: [[x, y], ...]} mapping into shape (T, K, 2)."""
    stacked = [np.asarray(data[k], dtype=np.float32) for k in key_order]
    return np.stack(stacked, axis=1)  # (T, K, 2)


def sequence_to_keypoint_dict(
    sequence: torch.Tensor,
    key_order: List[str],
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Dict[str, List[List[float]]]:
    """
    Convert a model output of shape (C, T) back to the JSON-friendly dict of lists.
    If mean/std are provided, the sequence is un-normalized before conversion.
    """
    # Accept (C, T), (1, C, T), or (C, T, 1) and squeeze singleton dims if present.
    if sequence.ndim == 3 and sequence.shape[0] == 1:
        sequence = sequence.squeeze(0)
    if sequence.ndim == 3 and sequence.shape[-1] == 1:
        sequence = sequence.squeeze(-1)
    if sequence.ndim != 2:
        raise ValueError(f"sequence must have shape (C, T); got {sequence.shape}")
    sequence = sequence.contiguous()
    if mean is not None and std is not None:
        # mean/std saved as (1, C, 1); reshape to (C, 1) for broadcasting over T
        mean_flat = mean.view(-1, 1).to(sequence.device, sequence.dtype)
        std_flat = std.view(-1, 1).to(sequence.device, sequence.dtype)
        sequence = sequence * std_flat + mean_flat
    sequence = sequence.permute(1, 0)  # (T, C)
    t, c = sequence.shape
    if c != len(key_order) * 2:
        raise ValueError(f"channel count {c} does not match key order of length {len(key_order)}")
    array = sequence.reshape(t, len(key_order), 2)
    return {k: array[:, idx].tolist() for idx, k in enumerate(key_order)}


class MotionSequenceDataset(Dataset):
    """
    Builds fixed-length clips from a single JSON motion file.
    The dataset pre-computes mean/std across all clips for normalization.
    """

    def __init__(
        self,
        json_path: str,
        window: int = 160,
        stride: int = 10,
        limit: Optional[int] = None,
        normalize: bool = True,
        seed: int = 0,
    ) -> None:
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"json path not found: {json_path}")

        with self.json_path.open("r") as f:
            data = json.load(f)

        self.key_order: List[str] = sorted(data.keys())
        self.array = _dict_to_array(data, self.key_order)  # (T, K, 2)
        self.window = window
        self.stride = stride
        self.normalize = normalize
        self.seed = seed
        self.num_joints = len(self.key_order)
        self.num_channels = self.num_joints * 2

        self.sequences: List[torch.Tensor] = self._make_sequences(limit)
        if len(self.sequences) == 0:
            raise ValueError("No sequences were generated. Adjust window/stride/limit.")

        if normalize:
            self.mean, self.std = self._compute_stats(self.sequences)
            self.sequences = [(seq - self.mean) / self.std for seq in self.sequences]
        else:
            # Broadcast shapes (1, C, 1) for simple arithmetic
            self.mean = torch.zeros((1, self.num_channels, 1), dtype=torch.float32)
            self.std = torch.ones((1, self.num_channels, 1), dtype=torch.float32)

    def _make_sequences(self, limit: Optional[int]) -> List[torch.Tensor]:
        total_frames = self.array.shape[0]
        starts = list(range(0, total_frames - self.window + 1, self.stride))
        if limit is not None and limit < len(starts):
            rng = np.random.default_rng(self.seed)
            starts = rng.choice(starts, size=limit, replace=False).tolist()

        sequences: List[torch.Tensor] = []
        for s in starts:
            clip = self.array[s : s + self.window]  # (window, K, 2)
            clip = torch.from_numpy(clip.reshape(self.window, -1).T).float()  # (C, window)
            sequences.append(clip)
        return sequences

    def _compute_stats(self, sequences: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(sequences, dim=0)  # (N, C, T)
        mean = stacked.mean(dim=(0, 2), keepdim=True)
        std = stacked.std(dim=(0, 2), keepdim=True).clamp(min=1e-6)
        return mean, std

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]

    def unnormalize(self, seq: torch.Tensor) -> torch.Tensor:
        """Undo normalization for a sequence shaped (C, T)."""
        return seq * self.std + self.mean

    @property
    def sequence_length(self) -> int:
        return self.window


__all__ = [
    "MotionSequenceDataset",
    "sequence_to_keypoint_dict",
]
