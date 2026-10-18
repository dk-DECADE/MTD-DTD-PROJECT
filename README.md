# 64×64 数字 6 去噪

`part1/CNNtrain.py` 是简化训练代码，`part1/app.py` 是网页入口。
训练一次并保存 `model.pt` 后，网页只加载模型做预测。

## 用 Docker 打开网页

1. 启动 **Docker Desktop**，等待显示引擎已运行；使用 Linux 容器。
2. 在 PowerShell 中进入项目并训练模型：

```powershell
cd C:\Users\decade\Desktop\DTD_TEST\part1
E:\Anaconda3\envs\pytorch\python.exe -X utf8 CNNtrain.py
```

训练完成后，同目录应出现 `model.pt` 和 `simple_comparison.png`。
修改训练代码中的 `alpha` 或 CNN 结构后，需要重新训练；若修改网络结构，也要同步修改 app.py。

3. 构建 Docker 镜像（首次需要联网下载 Python、CPU 版 PyTorch 和依赖）：

```powershell
docker build -t diffusion-six .
```

4. 启动容器：

```powershell
docker run -d --name diffusion-six -p 8501:8501 diffusion-six
```

5. 浏览器访问 **http://localhost:8501**。

网页可以使用示例图片，也可以上传干净的数字 6。白底黑字的上传图片请勾选反色，
点击“加噪并重建”，查看原图、含噪图、线性回归与 CNN 重建及 MSE。
本模型固定使用训练时的 alpha。它只在示例数字 6 上训练，其他手写图像的效果可能较差。

## 常用命令

```powershell
# 查看容器和日志
docker ps
docker logs diffusion-six

# 停止容器
docker stop diffusion-six

# 再次启动已有容器，无需重新创建
docker start diffusion-six
```

若网页代码或模型有更新，重新构建镜像并更换旧容器：

```powershell
docker build -t diffusion-six .
docker stop diffusion-six
docker rm diffusion-six
docker run -d --name diffusion-six -p 8501:8501 diffusion-six
```

如果提示无法连接 Docker 引擎，先检查 Docker Desktop 是否已启动。
如果本机 8501 端口被占用，可将启动命令中的 `-p 8501:8501` 改为 `-p 8502:8501`，
然后访问 http://localhost:8502。

## 不使用 Docker 的本地运行方式

```powershell
cd C:\Users\decade\Desktop\DTD_TEST\part1
E:\Anaconda3\envs\pytorch\python.exe -m pip install -r requirements.txt
E:\Anaconda3\envs\pytorch\python.exe -m streamlit run app.py
```

## 运行范围

以上步骤在自己的电脑上运行 Docker 网页。让同一局域网内其他人访问，需使用本机局域网 IP
（例如 http://192.168.1.10:8501）并允许防火墙入站访问 8501。
要让互联网用户访问，需要在有公网访问能力的服务器或容器平台上运行该镜像，并配置网络入口。
仅在本机启动 Docker，不会自动生成公网网址。

官方参考：[Streamlit Docker 部署](https://docs.streamlit.io/deploy/tutorials/docker)、
[Docker Desktop 端口映射](https://docs.docker.com/desktop/features/networking/)。
## U-Net 训练

新增 `part1/UNettrain.py`，按顺序完成读图、加噪、线性回归、U-Net 训练、模型保存和重建比较。
小型 U-Net 使用两层编码器、两层解码器和跳跃连接；输入和预测噪声都是 64×64。
有 CUDA 时自动使用 GPU，否则使用 CPU。不需要安装新的依赖。

```powershell
cd C:\Users\decade\Desktop\DTD_TEST\part1
E:\Anaconda3\envs\pytorch\python.exe -X utf8 UNettrain.py
```

默认参数为 `alpha=0.8`、`epochs=50`、`batch_size=16`，可以在脚本中修改。
使用同一张干净图生成 256 张训练含噪图和 64 张独立测试含噪图，随机种子与 CNN 脚本一致。
训练固定轮数，不根据测试集挑选模型。

输出文件：

- `unet_model.pt`：U-Net 参数及训练配置，单独保存。
- `unet_comparison.png`：原图、含噪图、线性回归、已有 CNN 和 U-Net 重建对比。
- `unet_metrics.csv`：整个测试集的平均重建 MSE，越小越好。

如果已有 `model.pt`，且其 alpha 与本次训练相同，会加载已有 CNN 在同一批测试含噪图上评估。
CNN 不会重新训练；对比时需确保 CNN 原先使用了同一张干净图和可比的训练设置。
如果没有 CNN 模型，或 alpha 不匹配，则只比较线性回归和 U-Net。

MSE 使用未裁剪的重建结果计算，图片统一按 [0,1] 显示。结果仅针对同一张数字 6 的新噪声，
不代表对新的手写图像有相同效果。当前 Docker 网页仍加载 CNN 的 `model.pt`；本次新增的是 U-Net 训练。
