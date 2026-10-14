"""网页入口：加载训练好的模型，上传图片后加噪和重建。"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
import torch
from torch import nn


folder = Path(__file__).resolve().parent
torch.set_num_threads(2)
st.set_page_config(page_title="数字 6 去噪实验", page_icon="6", layout="wide")
st.title("64×64 数字 6 去噪实验")
st.write("使用示例图片或上传数字 6，比较线性回归和 CNN 的重建效果。")


@st.cache_resource
def load_model():
    # 结构必须与 CNNtrain.py 中的网络一致；加载时不会重新训练。
    cnn = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, kernel_size=3, padding=1),
    )
    saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    cnn.load_state_dict(saved["cnn_weights"])
    cnn.eval()
    return cnn, saved


if not (folder / "model.pt").is_file():
    st.error("没有找到 model.pt，请先运行 CNNtrain.py 训练并保存模型。")
    st.stop()

cnn, saved = load_model()
alpha = saved["alpha"]
st.caption(f"训练及网页使用的 alpha = {alpha}；所有输入转换为 64×64 灰度图。")

uploaded = st.file_uploader("上传一张干净的数字 6（不上传则使用示例）", type=["png", "jpg", "jpeg"])
invert = st.checkbox("上传图片是白底黑字，需要反色", value=False)
seed = st.number_input("噪声随机种子（修改后可以体验不同噪声）", min_value=0,
                       max_value=1000000, value=43, step=1)

try:
    source = uploaded if uploaded is not None else folder / "clean_six.png"
    with Image.open(source) as original:
        clean_image = ImageOps.exif_transpose(original).convert("L")
        if uploaded is not None and invert:
            clean_image = ImageOps.invert(clean_image)
        clean_image = ImageOps.pad(clean_image, (64, 64),
                                  method=Image.Resampling.LANCZOS, color=0)
        clean_array = np.asarray(clean_image, dtype=np.float32) / 255.0
except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
    st.error("图片无法读取，请上传有效的 PNG 或 JPG 图片。")
    st.stop()

st.image(clean_array, caption="当前干净输入图片", width=192, clamp=True)

if st.button("加噪并重建", type="primary"):
    clean = torch.from_numpy(clean_array.copy())[None, None]
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(int(seed)))
    signal_scale = alpha ** 0.5
    noise_scale = (1 - alpha) ** 0.5
    noisy = signal_scale * clean + noise_scale * noise
    predicted_linear = saved["linear_w"] * noisy + saved["linear_b"]
    with torch.no_grad():
        predicted_cnn = cnn(noisy)
    linear = (noisy - noise_scale * predicted_linear) / signal_scale
    reconstructed = (noisy - noise_scale * predicted_cnn) / signal_scale

    images = [clean, noisy, linear, reconstructed]
    titles = ["干净原图", "含噪图", "线性回归重建", "CNN 重建"]
    for column, title, image in zip(st.columns(4), titles, images):
        column.image(image[0, 0].numpy().clip(0, 1), caption=title, width=224)
        column.metric("与原图的 MSE", f"{((image - clean) ** 2).mean().item():.6f}")
    st.caption("MSE 越小越好。MSE 使用未裁剪像素计算，图片显示范围为 [0,1]。")

st.info("模型只在提供的示例数字 6 上训练。上传其他手写图像的效果可能较差。")

st.divider()
st.subheader("项目团队")
st.image(str(folder / "teammates.jpg"), caption="Teammates", width=640)
