"""Alpha-conditioned CNN and portable model loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from diffusion_six.core import SIZE, validate_alpha, validate_image


class NoiseCNN(nn.Module):
    """Predict epsilon from the noisy image and a broadcast alpha channel."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=4, dilation=4), nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, noisy: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        alpha_map = alpha.reshape(-1, 1, 1, 1).expand_as(noisy)
        return self.network(torch.cat((noisy, alpha_map), dim=1))


def load_model(path: Path | str = "model.pt") -> tuple[NoiseCNN, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2 or checkpoint.get("metadata", {}).get("image_size") != [SIZE, SIZE]:
        raise ValueError("Unsupported model checkpoint format. Retrain with train.py.")
    model = NoiseCNN()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint["metadata"]


def predict_noise(model: NoiseCNN, noisy: np.ndarray, alpha: float) -> np.ndarray:
    noisy = validate_image(noisy)
    alpha = validate_alpha(alpha)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(noisy.copy())[None, None],
                           torch.tensor([alpha], dtype=torch.float32))
    return prediction[0, 0].numpy()


def predict_baseline(path: Path | str, noisy: np.ndarray, alpha: float) -> np.ndarray:
    noisy = validate_image(noisy)
    alpha = validate_alpha(alpha)
    with np.load(path, allow_pickle=False) as baseline:
        features = np.stack((noisy.reshape(-1), np.full(SIZE * SIZE, alpha),
                             np.ones(SIZE * SIZE)), axis=-1)
        result = np.sum(baseline["coef"] * features, axis=-1)
    return result.reshape(SIZE, SIZE).astype(np.float32)
