代码文件：`CNNtrain.py`，按顺序完成：读图 → 加噪 → 线性回归 → CNN → 重建 → 比较 MSE。

运行

本机已有 PyTorch 环境，直接运行：

```powershell
cd C:\Users\decade\Desktop\DTD_TEST\part1
E:\Anaconda3\envs\pytorch\python.exe part1_simple.py
```

其他电脑使用 Python 3.10 或以上版本：

```powershell
python -m pip install -r requirements.txt
python part1_simple.py
```

可修改

`alpha = 0.8`：越接近 1，噪声越少，取值必须在 0 和 1 之间。
`epochs = 50`：CNN 训练轮数。
`batch_size = 16`：每次训练使用的图片数量。

图片与结果

输入图片为 `clean_six.png`，请保留这张图片。可以将它替换为自己的黑底白字数字 6，代码会转换为 64×64 灰度图。

运行后在终端显示线性回归和 CNN 的测试集平均 MSE（越小越好），并保存对比图 `simple_comparison.png`。