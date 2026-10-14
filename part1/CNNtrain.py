"""Part 1 入门版：按顺序阅读下面 6 个步骤即可。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn

alpha = 0.8       # 保留的信号比例：越接近 1，噪声越少；必须在 0 和 1 之间。
epochs = 50      # CNN 训练轮数。
batch_size = 16  # 每次训练使用多少张含噪图。

torch.manual_seed(42)
torch.set_num_threads(2)
folder = Path(__file__).resolve().parent

# =====读取数字 6=====
with Image.open(folder / "clean_six.png") as image:
    image = image.convert("L").resize((64, 64))
    clean_array = np.asarray(image, dtype=np.float32) / 255.0

clean = torch.from_numpy(clean_array)[None, None]

# =====按公式加噪=====
# z = sqrt(alpha) * x + sqrt(1-alpha) * noise
# 同一张干净图生成 256 张训练含噪图、64 张测试含噪图。
# 两组噪声分别生成，测试噪声不会用于训练。
train_generator = torch.Generator().manual_seed(42)
test_generator = torch.Generator().manual_seed(43)
noise_train = torch.randn(256, 1, 64, 64, generator=train_generator)
noise_test = torch.randn(64, 1, 64, 64, generator=test_generator)
signal_scale = alpha ** 0.5
noise_scale = (1 - alpha) ** 0.5
noisy_train = signal_scale * clean + noise_scale * noise_train
noisy_test = signal_scale * clean + noise_scale * noise_test

# ===== 3. 训练线性回归基线 =====
# 让模型学习：预测噪声 = w * 含噪像素 + b。
# polyfit(..., deg=1) 就是拟合一条直线，得到斜率 w 和截距 b。
# 所有像素共享这条直线，因此它看不到相邻像素的结构。
w, b = np.polyfit(noisy_train.numpy().ravel(), noise_train.numpy().ravel(), deg=1)
predicted_noise_linear = float(w) * noisy_test + float(b)

# ===== 4. 训练小型 CNN =====
# CNN 输入一张含噪图，输出一张同尺寸的预测噪声图。
# 3×3 卷积可以观察周围的像素；padding=1 用来保持 64×64 尺寸。
cnn = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 1, kernel_size=3, padding=1),
)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_function = nn.MSELoss()

cnn.train()
for epoch in range(epochs):
    order = torch.randperm(len(noisy_train))
    total_loss = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        prediction = cnn(noisy_train[indices])
        loss = loss_function(prediction, noise_train[indices])
        optimizer.zero_grad()  # 清除上一步的梯度。
        loss.backward()        # 计算如何调整模型参数。
        optimizer.step()       # 更新模型参数。
        total_loss += loss.item() * len(indices)
    if (epoch + 1) % 5 == 0:
        print(f"训练轮数 {epoch + 1}/{epochs}，噪声预测损失 {total_loss / len(order):.6f}",
              flush=True)

# ===== 5. 用预测噪声重建干净图 =====
# 将加噪公式反过来：x_hat = (z - sqrt(1-alpha)*预测噪声) / sqrt(alpha)。
cnn.eval()
# 保存训练好的 CNN 和线性回归参数，网页直接加载，不需要再次训练。
torch.save({"cnn_weights": cnn.state_dict(), "alpha": float(alpha),
            "linear_w": float(w), "linear_b": float(b)}, folder / "model.pt")
print(f"模型已保存：{folder / 'model.pt'}")
with torch.no_grad():
    predicted_noise_cnn = cnn(noisy_test)
restored_linear = (noisy_test - noise_scale * predicted_noise_linear) / signal_scale
restored_cnn = (noisy_test - noise_scale * predicted_noise_cnn) / signal_scale

# ===== 6. 比较 MSE 和重建图 =====
# MSE 是重建图与干净图之间的平均平方误差，越小越好。
# 两个模型使用完全相同的 64 张测试含噪图，以下是整个测试集的平均 MSE。
mse_linear = ((restored_linear - clean) ** 2).mean().item()
mse_cnn = ((restored_cnn - clean) ** 2).mean().item()
print(f"\n线性回归 MSE：{mse_linear:.6f}")
print(f"CNN      MSE：{mse_cnn:.6f}")
print("线性回归只读取单个像素；CNN 可以利用邻域中的笔画和背景信息。")
print("注意：这里只测试同一张数字 6 的新噪声，不能代表对其他手写图像的效果。")

# 展示第一张测试图，不根据结果好坏挑选样本；标题中的 MSE 是整个测试集平均值。
images = [clean[0, 0], noisy_test[0, 0], restored_linear[0, 0], restored_cnn[0, 0]]
titles = ["Clean 64x64", f"Noisy (alpha={alpha})",
          f"Linear\nTest MSE={mse_linear:.5f}", f"CNN\nTest MSE={mse_cnn:.5f}"]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
for ax, image, title in zip(axes, images, titles):
    ax.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(folder / "simple_comparison.png", dpi=150)
plt.close(fig)
print(f"对比图已保存：{folder / 'simple_comparison.png'}")
