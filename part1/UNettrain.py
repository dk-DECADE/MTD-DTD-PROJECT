"""64×64 数字 6 去噪：小型 U-Net，含编码器、解码器和跳跃连接。"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn


class UNet(nn.Module):
    """输入和输出均为 [图片数量, 1, 64, 64]，输出预测的噪声。"""

    def __init__(self):
        super().__init__()

        # 每个模块连续做两次卷积，padding=1 保持宽高不变。
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
            )

        # 编码器：64×64 → 32×32 → 16×16，逐渐提取较大范围的结构。
        self.encoder1 = double_conv(1, 8)
        self.encoder2 = double_conv(8, 16)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(16, 32)

        # 解码器：16×16 → 32×32 → 64×64，恢复图像分辨率。
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder2 = double_conv(32, 16)
        self.up1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.decoder1 = double_conv(16, 8)
        self.output = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, noisy):
        skip1 = self.encoder1(noisy)                  # [N, 8, 64, 64]
        skip2 = self.encoder2(self.pool(skip1))       # [N, 16, 32, 32]
        middle = self.bottleneck(self.pool(skip2))   # [N, 32, 16, 16]

        # 跳跃连接：拼接编码器的特征，帮助解码器恢复笔画细节。
        decoded2 = self.decoder2(torch.cat((self.up2(middle), skip2), dim=1))
        decoded1 = self.decoder1(torch.cat((self.up1(decoded2), skip1), dim=1))
        return self.output(decoded1)


# 这个判断让其他程序可以 import UNet，而不会自动开始训练。
if __name__ == "__main__":
    # 可以修改这三个参数；默认与现有 CNN 训练使用相同设置。
    alpha = 0.8
    epochs = 50
    batch_size = 16

    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = Path(__file__).resolve().parent
    print(f"训练设备：{device}；图像尺寸：64×64", flush=True)

    # ===== 1. 读取干净图像 =====
    with Image.open(folder / "clean_six.png") as image:
        clean_array = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
    clean = torch.from_numpy(clean_array)[None, None]

    # ===== 2. 生成训练和测试含噪图，公式与 CNN 相同 =====
    train_generator = torch.Generator().manual_seed(42)
    test_generator = torch.Generator().manual_seed(43)
    noise_train = torch.randn(256, 1, 64, 64, generator=train_generator)
    noise_test = torch.randn(64, 1, 64, 64, generator=test_generator)
    signal_scale = alpha ** 0.5
    noise_scale = (1 - alpha) ** 0.5
    noisy_train = signal_scale * clean + noise_scale * noise_train
    noisy_test = signal_scale * clean + noise_scale * noise_test

    # ===== 3. 拟合逐像素线性回归基线 =====
    w, b = np.polyfit(noisy_train.numpy().ravel(), noise_train.numpy().ravel(), deg=1)
    predicted_linear = float(w) * noisy_test + float(b)
    restored_linear = (noisy_test - noise_scale * predicted_linear) / signal_scale

    # ===== 4. 训练 U-Net 预测噪声 =====
    unet = UNet().to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    unet.train()
    for epoch in range(epochs):
        order = torch.randperm(len(noisy_train))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            prediction = unet(noisy_train[indices].to(device))
            loss = loss_function(prediction, noise_train[indices].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"训练轮数 {epoch + 1}/{epochs}，噪声预测损失 {total_loss / len(order):.6f}",
                  flush=True)

    # ===== 5. 保存模型并重建测试图 =====
    unet.eval()
    torch.save({"unet_weights": {key: value.cpu() for key, value in unet.state_dict().items()},
                "alpha": float(alpha), "image_size": 64, "epochs": epochs,
                "linear_w": float(w), "linear_b": float(b)}, folder / "unet_model.pt")
    # 分批预测，减少一次性占用的显存。
    predictions = []
    with torch.no_grad():
        for start in range(0, len(noisy_test), batch_size):
            predictions.append(unet(noisy_test[start:start + batch_size].to(device)).cpu())
    predicted_unet = torch.cat(predictions)
    restored_unet = (noisy_test - noise_scale * predicted_unet) / signal_scale

    # ===== 6. 比较线性回归、已有 CNN 和 U-Net =====
    results = {"Linear": restored_linear}
    if (folder / "model.pt").is_file():
        # 只加载已有 CNN，不导入会自动训练的 CNNtrain.py。
        saved_cnn = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        if abs(saved_cnn["alpha"] - alpha) < 1e-6:
            cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
            )
            cnn.load_state_dict(saved_cnn["cnn_weights"])
            cnn.eval()
            with torch.no_grad():
                predicted_cnn = cnn(noisy_test)
            results["CNN (saved)"] = (noisy_test - noise_scale * predicted_cnn) / signal_scale
        else:
            print("已有 CNN 的 alpha 不同，本次不将它加入对比。")
    results["U-Net"] = restored_unet

    rows = []
    print("\n64 张独立测试含噪图的平均 MSE（越小越好）：")
    for name, restored in results.items():
        mse = ((restored - clean) ** 2).mean().item()
        rows.append({"model": name, "alpha": alpha, "test_samples": len(noisy_test), "mse": mse})
        print(f"{name:<12} {mse:.6f}")
    with (folder / "unet_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 固定展示第一个测试样本，标题中的 MSE 是整个测试集的平均值。
    images = [clean[0, 0], noisy_test[0, 0]] + [image[0, 0] for image in results.values()]
    titles = ["Clean 64x64", f"Noisy (alpha={alpha})"]
    titles += [f"{row['model']}\nTest MSE={row['mse']:.5f}" for row in rows]
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3.5))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "unet_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n模型已保存：{folder / 'unet_model.pt'}")
    print(f"对比图已保存：{folder / 'unet_comparison.png'}")
    print("CNN 是已有模型的测试结果，没有在本脚本中重新训练。")
    print("MSE 使用未裁剪像素计算；这里只测试同一张数字 6 的新噪声。")
