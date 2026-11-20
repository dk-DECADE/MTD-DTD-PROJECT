"""测试去噪效果，并保存权重、指标和报告配图。"""

import hashlib
import json
import platform
import time

import matplotlib
import numpy as np
import sklearn
import torch
from PIL import Image

from diffusion_six.core import ALPHA_MAX, ALPHA_MIN, SIZE
from diffusion_six.data import evaluation_batch
from diffusion_six.nist import PARTITIONS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEST_SEED = 12345
TEST_REPEATS = 8
TEST_ALPHAS = (0.15, 0.35, 0.55, 0.75, 0.95)


def restore(noisy, predicted_noise, alpha):
    alpha = alpha[:, None, None]
    clean = (noisy - np.sqrt(1 - alpha) * predicted_noise) / np.sqrt(alpha)
    return clean.clip(0, 1)


def image_mse(predicted, clean):
    return float(np.mean((predicted.astype(np.float64) - clean) ** 2))


def evaluate(model, baseline, batch):
    clean, noisy, alpha, _ = batch
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(noisy), 32):
            stop = start + 32
            noisy_tensor = torch.from_numpy(noisy[start:stop, None])
            alpha_tensor = torch.from_numpy(alpha[start:stop])
            predictions.append(model(noisy_tensor, alpha_tensor)[:, 0].numpy())

    cnn_images = restore(noisy, np.concatenate(predictions), alpha)
    linear_images = restore(noisy, baseline.predict(noisy, alpha), alpha)
    metrics = {
        "noisy_mse": image_mse(noisy, clean),
        "linear_mse": image_mse(linear_images, clean),
        "cnn_mse": image_mse(cnn_images, clean),
    }
    metrics["cnn_improvement_percent"] = 100 * (1 - metrics["cnn_mse"] / metrics["linear_mse"])
    return metrics, (clean, noisy, linear_images, cnn_images)


def model_metadata(settings, splits, best_epoch, history, baseline_samples):
    info = {
        "seed": settings.seed,
        "epochs": settings.epochs,
        "best_epoch": best_epoch,
        "samples_per_epoch": settings.samples_per_epoch,
        "baseline_samples": baseline_samples,
        "alpha_range": [ALPHA_MIN, ALPHA_MAX],
        "image_size": [SIZE, SIZE],
        "source_image_size": [8, 8],
        "resize": "bilinear before independent 64x64 noise",
        "baseline": "independent pixelwise linear regression on noisy pixel, alpha and intercept",
        "dataset": "sklearn bundled UCI optical recognition digits, label 6, divided by 16, resized to 64x64",
        "split_ids": {name: split["ids"] for name, split in splits.items()},
        "split_counts": {name: len(split["ids"]) for name, split in splits.items()},
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": settings.device,
        "validation_mse": history[best_epoch - 1]["validation_reconstruction_mse"],
        "evaluation_seed": TEST_SEED,
        "evaluation_repeats": TEST_REPEATS,
    }
    if not settings.digit_six:
        info.update(
            dataset="NIST Special Database 19 (2nd ed.), by_class, digits 0–9",
            source_image_size=[128, 128],
            resize="invert, crop ink, scale with Lanczos to fit 56x56, center on 64x64 black canvas",
            partitions={name: list(groups) for name, groups in PARTITIONS.items()},
            digits=list(range(10)),
        )
    return info


def digit_metrics(test_labels, images):
    clean, _, linear_images, cnn_images = images
    labels = np.repeat(test_labels, TEST_REPEATS)
    results = []
    for digit in range(10):
        selected = labels == digit
        results.append({
            "digit": digit,
            "observations": int(selected.sum()),
            "linear_mse": image_mse(linear_images[selected], clean[selected]),
            "cnn_mse": image_mse(cnn_images[selected], clean[selected]),
        })
    return results


def save_examples(test, output, digit_six):
    if digit_six:
        samples = [0, 1, 2]
        examples = [(f"six_{index + 1}.png", index) for index in samples]
    else:
        first_image = [int(np.flatnonzero(test["labels"] == digit)[0]) for digit in range(10)]
        examples = [(f"digit-{digit}.png", index) for digit, index in enumerate(first_image)]
        samples = [first_image[digit] for digit in (0, 6, 9)]

    examples_dir = output / "examples"
    examples_dir.mkdir(exist_ok=True)
    for filename, index in examples:
        pixels = np.rint(test["images"][index] * 255).astype(np.uint8)
        Image.fromarray(pixels).save(examples_dir / filename)
    return [index * TEST_REPEATS for index in samples]


def plot_comparison(path, images, indices, digit_six):
    titles = ("Clean", "Noisy (display clipped)", "Linear", "CNN")
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for row, index in enumerate(indices):
        for column, (title, batch) in enumerate(zip(titles, images)):
            ax = axes[row, column]
            ax.imshow(batch[index], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if column > 0:
                error = np.mean((batch[index] - images[0][index]) ** 2)
                title = f"{title}\nMSE {error:.4f}"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    subject = "digit 6" if digit_six else "NIST digits 0, 6 and 9"
    fig.suptitle(f"Held-out {subject} reconstructions | seed {TEST_SEED}", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_evaluation(path, history, by_alpha, best_epoch):
    fig, (validation, test) = plt.subplots(1, 2, figsize=(10, 3.8))
    validation.plot([row["epoch"] for row in history],
                    [row["validation_reconstruction_mse"] for row in history])
    validation.axvline(best_epoch, color="#64748b", linestyle="--", label=f"Selected epoch {best_epoch}")
    validation.set(xlabel="Epoch", ylabel="Validation reconstruction MSE", title="Checkpoint selection")
    validation.legend()

    alphas = [row["alpha"] for row in by_alpha]
    test.plot(alphas, [row["linear_mse"] for row in by_alpha], marker="o", label="Linear")
    test.plot(alphas, [row["cnn_mse"] for row in by_alpha], marker="o", label="CNN")
    test.set(xlabel="Alpha (higher = less noise)", ylabel="Test reconstruction MSE",
             title="Same held-out images and noise")
    test.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_results(settings, splits, model, baseline, best_epoch, history, baseline_samples, started):
    output = settings.output_dir
    test = splits["test"]
    test_images = test["images"]
    test_batch = evaluation_batch(test_images, repeats=TEST_REPEATS, seed=TEST_SEED)
    overall, images = evaluate(model, baseline, test_batch)
    by_alpha = []
    for alpha in TEST_ALPHAS:
        test_batch = evaluation_batch(test_images, repeats=TEST_REPEATS, seed=TEST_SEED, alpha=alpha)
        metrics, _ = evaluate(model, baseline, test_batch)
        by_alpha.append({"alpha": alpha, **metrics})

    results_dir = output / "results"
    results_dir.mkdir(exist_ok=True)
    indices = save_examples(test, output, settings.digit_six)
    plot_comparison(results_dir / "comparison.png", images, indices, settings.digit_six)
    plot_evaluation(results_dir / "evaluation.png", history, by_alpha, best_epoch)

    metadata = model_metadata(settings, splits, best_epoch, history, baseline_samples)
    model_path = output / "model.pt"
    baseline_path = output / "baseline.npz"
    torch.save({"format_version": 2, "state_dict": model.state_dict(), "metadata": metadata}, model_path)
    baseline.save(baseline_path)
    summary = {
        "metadata": metadata,
        "test": overall,
        "by_alpha": by_alpha,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "training_seconds": time.perf_counter() - started,
    }
    if not settings.digit_six:
        summary["by_digit"] = digit_metrics(test["labels"], images)
    (results_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return overall
