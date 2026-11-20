"""A per-pixel linear noise predictor, with no nonlinear or spatial features."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from diffusion_six.core import SIZE


class LinearNoisePredictor:
    """At pixel j: epsilon_j = w_j z_j + v_j alpha + b_j.

    Independent least squares at each pixel avoids a 4096 x 4096 dense model.
    A small ridge penalty regularizes signal/alpha coefficients only.
    """

    def fit(self, noisy: np.ndarray, alpha: np.ndarray, noise: np.ndarray):
        z = noisy.reshape(len(noisy), -1).astype(np.float64)
        eps = noise.reshape(len(noise), -1).astype(np.float64)
        a = alpha.astype(np.float64)
        gram = np.zeros((SIZE * SIZE, 3, 3), dtype=np.float64)
        gram[:, 0, 0] = np.sum(z * z, axis=0) + 1.0
        gram[:, 0, 1] = gram[:, 1, 0] = np.sum(z * a[:, None], axis=0)
        gram[:, 0, 2] = gram[:, 2, 0] = z.sum(axis=0)
        gram[:, 1, 1] = np.sum(a * a) + 1.0
        gram[:, 1, 2] = gram[:, 2, 1] = a.sum()
        gram[:, 2, 2] = len(a)
        rhs = np.stack((np.sum(z * eps, axis=0), np.sum(a[:, None] * eps, axis=0),
                        eps.sum(axis=0)), axis=-1)
        self.coef = np.linalg.solve(gram, rhs[..., None])[..., 0]
        return self

    def predict(self, noisy: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        z = noisy.reshape(len(noisy), -1)
        return (z * self.coef[None, :, 0] + alpha[:, None] * self.coef[None, :, 1]
                + self.coef[None, :, 2]).reshape(-1, SIZE, SIZE).astype(np.float32)

    def save(self, path: Path | str):
        np.savez(path, coef=self.coef, image_size=np.array([SIZE, SIZE]))

    @classmethod
    def load(cls, path: Path | str):
        with np.load(path, allow_pickle=False) as data:
            if data["image_size"].tolist() != [SIZE, SIZE]:
                raise ValueError("Baseline image size does not match this project.")
            model = cls()
            model.coef = data["coef"].copy()
        return model
