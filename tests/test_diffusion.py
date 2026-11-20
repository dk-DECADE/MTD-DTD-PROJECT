from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from diffusion_six.core import SIZE, add_noise, load_image, mse, png_bytes, reconstruct
from diffusion_six.data import clean_splits
from diffusion_six.model import NoiseCNN, load_model, predict_baseline, predict_noise

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def model():
    torch.set_num_threads(2)
    return load_model(ROOT / "model.pt")[0]


@pytest.fixture
def clean():
    return load_image(ROOT / "examples" / "six_1.png")


@pytest.mark.parametrize("alpha", [0.15, 0.55, 0.95, 1.0])
def test_true_noise_exactly_recovers_clean_image(clean, alpha):
    noisy, noise = add_noise(clean, alpha, seed=12)
    restored = reconstruct(noisy, noise, alpha, clip=False)
    assert noisy.shape == noise.shape == restored.shape == (64, 64)
    np.testing.assert_allclose(restored, clean, atol=1e-6)


def test_noise_is_reproducible_and_unclipped(clean):
    first, _ = add_noise(clean, 0.15, 42)
    second, _ = add_noise(clean, 0.15, 42)
    different, _ = add_noise(clean, 0.15, 43)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.min() < 0 and first.max() > 1


@pytest.mark.parametrize("alpha", [0, -0.1, 1.01, np.nan, np.inf])
def test_invalid_alpha_is_rejected(clean, alpha):
    with pytest.raises(ValueError, match="Alpha"):
        add_noise(clean, alpha)
    with pytest.raises(ValueError, match="Alpha"):
        reconstruct(clean, clean, alpha)


@pytest.mark.parametrize("invalid", [np.zeros((8, 8)), np.zeros((64, 64, 3)),
                                     np.full((64, 64), np.nan), np.full((64, 64), 2)])
def test_invalid_clean_image_is_rejected(invalid):
    with pytest.raises(ValueError):
        add_noise(invalid, 0.5)


def test_mse_has_known_value():
    assert mse(np.zeros((64, 64)), np.full((64, 64), 0.5)) == 0.25


def test_upload_preprocessing_and_polarity():
    canvas = np.full((180, 90), 255, dtype=np.uint8)
    canvas[30:140, 25:65] = 0
    data = BytesIO()
    Image.fromarray(canvas).save(data, format="PNG")
    clean = load_image(data.getvalue(), dark_strokes=True)
    assert clean.shape == (64, 64)
    assert clean[32, 32] > 0.95
    assert clean[0, 0] == 0
    assert clean.min() >= 0 and clean.max() <= 1


def test_transparent_upload_uses_correct_background():
    data = BytesIO()
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(data, format="PNG")
    assert np.max(load_image(data.getvalue(), dark_strokes=True)) == 0


def test_corrupt_upload_is_rejected():
    with pytest.raises((UnidentifiedImageError, OSError)):
        load_image(b"not an image")


def test_download_is_native_64_by_64(clean):
    with Image.open(BytesIO(png_bytes(clean))) as rendered:
        assert rendered.size == (64, 64)


def test_clean_splits_are_disjoint_and_repeatable():
    first, second = clean_splits(42), clean_splits(42)
    ids = [set(first[name]["ids"]) for name in ("train", "validation", "test")]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert len(set.union(*ids)) == 181
    for name in first:
        assert first[name]["ids"] == second[name]["ids"]
        assert first[name]["images"].shape[1:] == (64, 64)


def test_cnn_forward_preserves_native_resolution():
    torch.set_num_threads(2)
    network = NoiseCNN()
    output = network(torch.randn(2, 1, 64, 64), torch.tensor([0.15, 0.95]))
    assert output.shape == (2, 1, 64, 64)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("alpha", [0.15, 0.55, 0.95])
def test_saved_predictors_return_finite_bounded_reconstructions(model, clean, alpha):
    noisy, _ = add_noise(clean, alpha, 42)
    for predicted in (predict_noise(model, noisy, alpha),
                      predict_baseline(ROOT / "baseline.npz", noisy, alpha)):
        output = reconstruct(noisy, predicted, alpha)
        assert output.shape == (SIZE, SIZE)
        assert np.isfinite(output).all()
        assert output.min() >= 0 and output.max() <= 1


def test_checkpoint_declares_64_resolution():
    _, metadata = load_model(ROOT / "model.pt")
    assert metadata["image_size"] == [64, 64]
    assert metadata["source_image_size"] == [8, 8]


def test_old_8_resolution_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format_version": 1, "metadata": {"image_size": [8, 8]}}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_model(path)
