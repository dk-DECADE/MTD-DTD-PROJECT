# DTD 去噪网页操作教程

本教程适用于 `C:\Users\decade\Desktop\DTD_project`。当前网页版使用数字 6 的模型，项目团队合照显示在页面底部的 **Project team** 区域。

## 1 打开网页

先打开 **Docker Desktop**，等待 Docker 引擎启动。当前电脑已经有镜像和容器，打开 PowerShell 运行：

```powershell
docker start dtd-project-local
```

浏览器打开 [http://127.0.0.1:8502](http://127.0.0.1:8502)。如果网页已经打开，刷新即可。想查看容器状态时，运行：

```powershell
docker ps -a --filter name=dtd-project-local
```

看到 `Up` 表示容器在运行，`Exited` 表示已停止。如果找不到这个容器，按第 4 节构建并创建。

## 2 使用去噪功能

1. 左侧选 **Try an example** 使用内置样例，或选 **Upload your own** 上传一张数字 6 的 PNG 或 JPG。可用 `examples\manual\6.png` 练习上传。
2. 白底黑字的上传图片勾选 **Dark digit on a light background**。黑底白字就取消勾选。
3. 第一次可以用 **Signal retention α = 0.65** 和 **Noise seed = 42**，然后点击 **Reconstruct image**。调低 α 会加重噪声。比较两次结果时，固定种子可以让同一张图在相同设置下得到同样的噪声。
4. 页面依次显示原图、带噪图、线性回归重建和 CNN 重建。**MSE 越小，重建图越接近原图**。点击 **Download CNN reconstruction** 下载 64×64 PNG。
5. 滚动到页面底部即可查看团队合照。

## 3 替换团队合照

项目根目录已放有 `teammates.jpeg`。以后想换照片，把新照片另存为 **JPEG**，用同名文件替换它，再按第 4 节重建镜像。文件名保持一致，就不用改 Python。

网页在 `app.py` 的末尾加载图片。这里的 `width=680` 控制显示宽度，图片保持原来的比例。

```python
team_photo = ROOT / "teammates.jpeg"
if team_photo.is_file():
    st.divider()
    st.header("Project team")
    st.image(str(team_photo), width=680)
```

`Dockerfile` 中的这一行把它和应用一起复制到镜像里。如果换了文件名，这两个文件中的名字都要改成实际文件名。

```dockerfile
COPY --chown=appuser:appuser app.py model.pt baseline.npz teammates.jpeg ./
```

## 4 用 Docker 更新网页

改过 `app.py`、合照或模型文件后，按以下顺序操作。

1. 在项目目录构建新镜像。最后的 `.` 表示使用当前目录里的 `Dockerfile` 和文件。

   ```powershell
   cd C:\Users\decade\Desktop\DTD_project
   docker build -t dtd-project:local .
   ```

2. 等构建成功，再替换旧容器。如果 `docker ps -a` 里有 `dtd-project-local`，先停止并移除旧容器。状态已经是 `Exited` 时跳过 `docker stop`。如果这个容器从未创建过，跳过这两条。

   ```powershell
   docker stop dtd-project-local
   docker rm dtd-project-local
   ```

3. 启动新容器，再刷新网页。`8502` 是浏览器访问的端口，`8501` 是容器里 Streamlit 的端口。

   ```powershell
   docker run -d --name dtd-project-local -p 127.0.0.1:8502:8501 dtd-project:local
   ```

   打开 [http://127.0.0.1:8502](http://127.0.0.1:8502)。这个地址限本机访问，小组成员或老师需要可访问的部署地址。

## 5 遇到问题时

| 现象 | 怎么处理 |
|---|---|
| `failed to connect to the docker API` | 打开 Docker Desktop，等引擎启动。运行 `docker version`，看到 `Server` 后再试。 |
| `container name ... is already in use` | 只想打开原网页就用 `docker start dtd-project-local`。要更新就按照第 4 节替换旧容器。 |
| 网页打不开或启动报错 | 运行 `docker ps -a --filter name=dtd-project-local` 查看状态，再运行 `docker logs --tail 50 dtd-project-local` 看错误。 |
| 网页仍显示旧照片 | 确认改的是项目根目录的 `teammates.jpeg`，然后重新构建并创建容器。只重启原容器会继续使用它创建时的镜像。 |

如果只想在本机改网页并快速预览，可在项目目录的 PowerShell 运行 `./start.ps1`，然后打开 [http://localhost:8501](http://localhost:8501)。用完按 `Ctrl+C` 停止。

参考：[Streamlit 的 Docker 部署教程](https://docs.streamlit.io/deploy/tutorials/docker) · [Docker 的端口映射说明](https://docs.docker.com/get-started/docker-concepts/running-containers/publishing-ports/)
