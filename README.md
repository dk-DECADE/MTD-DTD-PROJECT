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
