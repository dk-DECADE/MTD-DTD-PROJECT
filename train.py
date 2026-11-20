import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from diffusion_six.baseline import LinearNoisePredictor
from diffusion_six.data import clean_splits, evaluation_batch, noisy_batch
from diffusion_six.model import NoiseCNN
from diffusion_six.nist import nist_splits
from diffusion_six.training_results import evaluate as evaluate  # scripts/evaluate.py 也使用这个函数
from diffusion_six.training_results import save_results


# 直接运行本文件时，用下面的设置训练 0–9。想改轮数或换保存目录，只改这里。
PROJECT_DIR = Path(__file__).resolve().parent
EPOCHS = 120
DATA_DIR = PROJECT_DIR / "by_class"
OUTPUT_DIR = PROJECT_DIR / "experiments" / "nist-60epochs"
SAMPLES_PER_EPOCH = 1024
TRAIN_PER_DIGIT = 200
EVAL_PER_DIGIT = 15  # 每个数字分别取这么多张验证图和测试图

SEED = 42
BATCH_SIZE = 32
BASELINE_SAMPLES = 2048


def read_settings():
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--nist-dir", type=Path, default=DATA_DIR, help="NIST by_class 文件夹")
    dataset.add_argument("--digit-six", action="store_true", help="使用 scikit-learn 的数字 6 数据")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--nist-train-per-digit", type=int, default=TRAIN_PER_DIGIT)
    parser.add_argument("--nist-eval-per-digit", type=int, default=EVAL_PER_DIGIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path)
    settings = parser.parse_args()

    if settings.digit_six:
        settings.epochs = 200 if settings.epochs is None else settings.epochs
        settings.samples_per_epoch = 512 if settings.samples_per_epoch is None else settings.samples_per_epoch
        settings.output_dir = settings.output_dir or PROJECT_DIR
        settings.device = settings.device or "cpu"
    else:
        settings.epochs = EPOCHS if settings.epochs is None else settings.epochs
        settings.samples_per_epoch = SAMPLES_PER_EPOCH if settings.samples_per_epoch is None else settings.samples_per_epoch
        settings.output_dir = settings.output_dir or OUTPUT_DIR
        settings.device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if settings.output_dir.exists():
            parser.error(f"保存目录已存在：{settings.output_dir}。请修改文件顶部的 OUTPUT_DIR 或使用 --output-dir。")
        if min(settings.nist_train_per_digit, settings.nist_eval_per_digit) < 1:
            parser.error("每个数字的训练、验证与测试图片数都必须大于 0。")

    if min(settings.epochs, settings.samples_per_epoch) < 1:
        parser.error("训练轮数和每轮样本数都必须大于 0。")
    if settings.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请选用 CPU。")
    return settings


def set_random_seed(seed, device):
    # 固定随机种子，方便在同一环境重跑实验。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_baseline(train_images, rng):
    _, noisy, alpha, noise = noisy_batch(train_images, BASELINE_SAMPLES, rng)
    return LinearNoisePredictor().fit(noisy, alpha, noise)


def validation_mse(model, validation):
    clean, noisy, alpha = validation
    model.eval()
    predicted_noise = []
    with torch.inference_mode():
        for start in range(0, len(noisy), BATCH_SIZE):
            stop = start + BATCH_SIZE
            predicted_noise.append(model(noisy[start:stop], alpha[start:stop]))
        predicted_noise = torch.cat(predicted_noise)
        # 用模型估计的噪声反推原图，再算重建误差。
        alpha = alpha.reshape(-1, 1, 1, 1)
        restored = (noisy - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
        return torch.nn.functional.mse_loss(restored.clamp(0, 1), clean).item()


def train_cnn(train_images, validation_images, settings, rng):
    device = settings.device
    model = NoiseCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 每一轮都用同一批验证图和噪声，这样各轮的误差可以直接比较。
    clean, noisy, alpha, _ = evaluation_batch(validation_images, seed=settings.seed + 1)
    validation = (
        torch.as_tensor(clean[:, None], device=device),
        torch.as_tensor(noisy[:, None], device=device),
        torch.as_tensor(alpha, device=device),
    )
    history = []
    best_mse = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(1, settings.epochs + 1):
        model.train()
        _, noisy, alpha, noise = noisy_batch(train_images, settings.samples_per_epoch, rng)
        noisy = torch.as_tensor(noisy[:, None], device=device)
        alpha = torch.as_tensor(alpha, device=device)
        noise = torch.as_tensor(noise[:, None], device=device)
        losses = []

        for start in range(0, len(noisy), BATCH_SIZE):
            stop = start + BATCH_SIZE
            optimizer.zero_grad(set_to_none=True)
            predicted_noise = model(noisy[start:stop], alpha[start:stop])
            loss = torch.nn.functional.mse_loss(predicted_noise, noise[start:stop])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        score = validation_mse(model, validation)
        train_loss = float(np.mean(losses))
        history.append({
            "epoch": epoch,
            "train_noise_mse": train_loss,
            "validation_reconstruction_mse": score,
        })
        if score < best_mse:
            best_mse = score
            best_epoch = epoch
            best_weights = {name: weight.detach().cpu().clone() for name, weight in model.state_dict().items()}
        print(f"第 {epoch}/{settings.epochs} 轮 | 训练噪声 MSE {train_loss:.5f} | 验证重建 MSE {score:.5f}", flush=True)

    # 取验证误差最低的一轮，用 CPU 保存权重，方便网页加载。
    if best_weights is None:
        raise ValueError("验证误差无有效数值，未能选出可用权重。")
    model = model.cpu()
    model.load_state_dict(best_weights)
    return model, best_epoch, history


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    settings = read_settings()
    started = time.perf_counter()
    set_random_seed(settings.seed, settings.device)

    # 1. 先划分原图，再分别加噪，避免同一张原图出现在训练集和测试集。
    if settings.digit_six:
        splits = clean_splits(settings.seed)
        dataset_name = "scikit-learn 数字 6"
    else:
        splits = nist_splits(settings.nist_dir, settings.seed,
                             settings.nist_train_per_digit, settings.nist_eval_per_digit)
        dataset_name = f"NIST 0–9 ({settings.nist_dir})"
    counts = {name: len(split["ids"]) for name, split in splits.items()}
    print(f"数据集：{dataset_name} | 设备：{settings.device} | 轮数：{settings.epochs}", flush=True)
    print(f"训练 {counts['train']} 张，验证 {counts['validation']} 张，测试 {counts['test']} 张", flush=True)
    settings.output_dir.mkdir(parents=True, exist_ok=settings.digit_six)

    # 2. 先用线性回归做基线，再训练 CNN，两个模型都预测噪声。
    rng = np.random.default_rng(settings.seed)
    train_images = splits["train"]["images"]
    baseline = train_baseline(train_images, rng)
    model, best_epoch, history = train_cnn(train_images, splits["validation"]["images"], settings, rng)

    # 3. 在独立测试集比较结果，并保存权重、样图和指标。
    metrics = save_results(settings, splits, model, baseline, best_epoch, history, BASELINE_SAMPLES, started)
    print(f"最佳轮次：{best_epoch} | 测试重建 MSE：线性 {metrics['linear_mse']:.5f}，CNN {metrics['cnn_mse']:.5f}")
    print(f"结果已保存到：{settings.output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"训练失败：{error}") from error
