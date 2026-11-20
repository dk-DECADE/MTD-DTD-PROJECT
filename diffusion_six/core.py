"""Numerical operations shared by training, inference and tests."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SIZE = 64
ALPHA_MIN = 0.15
ALPHA_MAX = 0.95


def validate_image(image: np.ndarray, *, clean: bool = False) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (SIZE, SIZE):
        raise ValueError(f"Expected a {SIZE} x {SIZE} image.")
    if not np.isfinite(array).all():
        raise ValueError("Image must contain finite numbers.")
    if clean and (array.min() < 0 or array.max() > 1):
        raise ValueError("Clean image pixels must be in [0, 1].")
    return array


def validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Alpha must be finite and in (0, 1].")
    return alpha


def add_noise(image: np.ndarray, alpha: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """z = sqrt(alpha) x + sqrt(1-alpha) epsilon; do not clip z."""
    image = validate_image(image, clean=True)
    alpha = validate_alpha(alpha)
    noise = np.random.default_rng(seed).standard_normal(image.shape).astype(np.float32)
    noisy = np.sqrt(alpha) * image + np.sqrt(1 - alpha) * noise
    return noisy.astype(np.float32), noise


def reconstruct(noisy: np.ndarray, predicted_noise: np.ndarray, alpha: float,
                *, clip: bool = True) -> np.ndarray:
    noisy = validate_image(noisy)
    predicted_noise = validate_image(predicted_noise)
    alpha = validate_alpha(alpha)
    result = (noisy - np.sqrt(1 - alpha) * predicted_noise) / np.sqrt(alpha)
    return np.clip(result, 0, 1) if clip else result


def mse(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = validate_image(reference)
    prediction = validate_image(prediction)
    return float(np.mean((reference.astype(np.float64) - prediction) ** 2))


def load_image(source: bytes | Path | str, *, dark_strokes: bool = False) -> np.ndarray:
    """Resize with aspect ratio preserved; normalize to light strokes on black."""
    data = BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(data) as original:
        if original.width * original.height > 16_000_000:
            raise ValueError("Please use an image smaller than 16 megapixels.")
        original = ImageOps.exif_transpose(original)
        background = (255, 255, 255, 255) if dark_strokes else (0, 0, 0, 255)
        rgba = original.convert("RGBA")
        composite = Image.alpha_composite(Image.new("RGBA", rgba.size, background), rgba)
        grayscale = composite.convert("L")
        if dark_strokes:
            grayscale = ImageOps.invert(grayscale)
        resized = ImageOps.pad(grayscale, (SIZE, SIZE), method=Image.Resampling.LANCZOS,
                               color=0, centering=(0.5, 0.5))
        return validate_image(np.asarray(resized, dtype=np.float32) / 255, clean=True)


def png_bytes(image: np.ndarray, scale: int = 1) -> bytes:
    image = validate_image(image, clean=True)
    rendered = Image.fromarray(np.rint(image * 255).astype(np.uint8))
    rendered = rendered.resize((SIZE * scale, SIZE * scale), Image.Resampling.NEAREST)
    buffer = BytesIO()
    rendered.save(buffer, format="PNG")
    return buffer.getvalue()
