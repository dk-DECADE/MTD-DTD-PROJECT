"""Offline sklearn digits data, split BEFORE generating any noisy observations."""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from diffusion_six.core import ALPHA_MAX, ALPHA_MIN, SIZE


def clean_splits(seed: int = 42) -> dict:
    digits = load_digits()
    indices = np.flatnonzero(digits.target == 6)
    train_ids, test_ids = train_test_split(indices, test_size=0.2, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=seed)
    # Source images have only 8 x 8 information. Resize BEFORE adding independent
    # 64 x 64 noise; no upsampling is performed after inference.
    images = np.stack([
        np.asarray(Image.fromarray(image.astype(np.float32) / 16).resize(
            (SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
        for image in digits.images
    ])
    return {name: {"images": images[ids], "ids": ids.tolist()}
            for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids))}


def noisy_batch(images: np.ndarray, count: int, rng: np.random.Generator,
                alpha: float | None = None) -> tuple[np.ndarray, ...]:
    clean = images[rng.integers(0, len(images), size=count)]
    alphas = (rng.uniform(ALPHA_MIN, ALPHA_MAX, size=count) if alpha is None
              else np.full(count, alpha)).astype(np.float32)
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    noisy = np.sqrt(alphas[:, None, None]) * clean + np.sqrt(1 - alphas[:, None, None]) * noise
    return clean, noisy, alphas, noise


def evaluation_batch(images: np.ndarray, *, repeats: int = 8, seed: int = 12345,
                     alpha: float | None = None) -> tuple[np.ndarray, ...]:
    """Every held-out image appears equally often with independent fixed noise."""
    clean = np.repeat(images, repeats, axis=0)
    rng = np.random.default_rng(seed)
    alphas = (rng.uniform(ALPHA_MIN, ALPHA_MAX, len(clean)) if alpha is None
              else np.full(len(clean), alpha)).astype(np.float32)
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    noisy = np.sqrt(alphas[:, None, None]) * clean + np.sqrt(1 - alphas[:, None, None]) * noise
    return clean, noisy, alphas, noise
