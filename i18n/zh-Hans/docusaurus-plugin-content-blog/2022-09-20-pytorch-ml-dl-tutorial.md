---
slug: pytorch-ml-dl-tutorial
title: "面向科研的 PyTorch 学习路线"
description: "从张量和训练循环，到迁移学习、模型评估、安全保存和部署，一条重视可复现性的 PyTorch 学习路线。"
authors: [liangchao]
category: 人工智能与机器学习
article_type: 技术指南
tags: [machine-learning, artificial-intelligence, python, computer-vision]
image: /img/blog-default.jpg
---

## 概述

这份路线图帮助 Python 用户从第一个 PyTorch 模型走到可复现、可核查的科研实验。全文围绕一个可运行的小例子展开，讲解科研中最关键的几件事：设备管理、随机种子、数据划分、评价指标、检查点、版本记录和验证。

- **读者：** 准备开展可复现机器学习实验的 Python 用户
- **版本：** PyTorch 2.x 和 torchvision 的近期版本；具体 API 以所安装版本的文档为准

<!-- truncate -->

## 1. 按官方说明安装

PyTorch 的安装包取决于操作系统和加速硬件。请按 [PyTorch — Start Locally](https://pytorch.org/get-started/locally/) 页面给出的命令安装，不要照抄旧教程里的 CUDA 下载地址。

学习和持续集成用 CPU 版即可：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Windows 下请用 PowerShell 或命令提示符对应的激活命令。需要 CUDA、ROCm、XPU 等后端时，用官方页面生成安装命令，并记下实际使用的命令。

记录环境：

```bash
python --version
python -m pip freeze > requirements-lock.txt
```

长期维护的项目，建议用依赖文件管理版本，并有意识地锁定和升级，而不是只保存一次 `pip freeze` 的结果。

## 2. 检查运行环境，统一选择设备

不要在代码各处直接写 `.cuda()`。在一个地方选好设备，再把模型和张量都放到这个设备上：

```python
import torch

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = select_device()

print("PyTorch:", torch.__version__)
print("Device:", device)
print("CUDA runtime:", torch.version.cuda)
```

设备、加速卡型号、驱动和运行时版本、包版本和计算精度，都要和实验结果一起记录。

## 3. 创建模型前先固定随机性

随机种子必须在初始化模型、打乱数据之前设置：

```python
import os
import random

import numpy as np
import torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True, warn_only=True)
```

固定种子可以提高可重复性，但在不同的 PyTorch 版本、设备、算子实现或分布式配置下，结果仍可能有细微差异。应报告可接受的误差范围，并对重要实验做重复。

## 4. 训练一个最小模型

XOR 例子不需要任何数据集，就能演示张量、模块、logits、损失函数、优化器和推理的完整流程：

```python
import torch
from torch import nn

torch.manual_seed(42)

X = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(500):
    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.eval()
with torch.inference_mode():
    probabilities = torch.sigmoid(model(X))
    predictions = (probabilities >= 0.5).to(torch.int32)

print(probabilities.squeeze())
print(predictions.squeeze())
```

二分类时，用 `BCEWithLogitsLoss` 直接接收原始 logits，不要先加 sigmoid 再用 `BCELoss`。输出的概率值会随初始化、包版本和底层算子变化，因此这里不给出“标准答案”。

## 5. 组织科研数据集

数据划分要在任何依赖数据的预处理之前完成：

1. 明确观测单元；
2. 按能防止信息泄漏的单元划分，例如植株、小区、田块、日期或受试者；
3. 归一化和特征变换只在训练集上拟合；
4. 验证集固定下来，只用于模型选择；
5. 测试集只在最终评估时使用一次。

在图像表型中，如果按单张图像随机划分，同一植株的几乎相同的视角可能同时出现在训练集和验证集中。按组划分得到的结果更可信。

自定义数据集应返回样本和标签，不要暗中修改全局状态：

```python
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

class ImageTableDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(Path(row["path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])
```

构建 `rows` 时检查路径和标签是否有效，并记录缺失或损坏样本是如何处理的。

## 6. 保持训练循环清晰

每个 epoch 的职责要明确：遍历一个数据加载器，更新模型，返回按样本数加权的指标。

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        loss_sum += loss.detach().item() * batch_size
        sample_count += batch_size

    return loss_sum / sample_count
```

验证时：

- 调用 `model.eval()`；
- 用 `torch.inference_mode()` 包裹推理；
- 不更新任何模型参数；
- 在全部样本上汇总指标；
- 需要做误差分析时，保存预测结果和样本编号。

不要根据测试集的表现来挑选模型。

## 7. 混合精度只在支持时开启

自动混合精度可以提高 CUDA 上的训练速度，但它是性能优化，不影响正确性。先跑通全精度的基线，再考虑开启：

```python
from contextlib import nullcontext

import torch

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

for inputs, targets in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)

    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )

    with precision_context:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

旧的 `torch.cuda.amp.autocast` 和 `torch.cuda.amp.GradScaler` 写法在当前文档中已经弃用。更改精度后，要留意损失、梯度和指标中是否出现 NaN 或溢出。

## 8. 计算机视觉从成熟的基线模型开始

torchvision 的权重枚举把预训练参数和对应的预处理绑定在一起：

```python
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 4)
```

请记录使用的权重、输入分辨率、图像变换、类别映射和微调策略。另外，只换一个输出层并不能得到 YOLO 这样的检测器：目标检测还需要目标编码、专门的损失函数、解码、非极大值抑制、评估方法和针对任务的训练。

对植物图像，学到的模型要和简单基线比较，并在真正影响结果的维度上分别评估：品种、生育期、传感器、田块、日期和光照。

## 9. 安全地保存模型

保存的是参数字典和元数据，而不是把整个 Python 对象序列化：

```python
from pathlib import Path

import torch

checkpoint_path = Path("checkpoints/model.pt")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "class_names": class_names,
    },
    checkpoint_path,
)
```

只加载可信来源的文件，并指定加载到的设备：

```python
checkpoint = torch.load(
    "checkpoints/model.pt",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state_dict"])
```

`weights_only=True` 会限制 `torch.load` 能加载的内容，但外部来源的模型文件仍应视为不可信，需要核对来源和完整性。

## 10. 评估不能只看一个数字

在看到最终结果之前，就要先确定评价指标：

- **分类：** 混淆矩阵、各类精确率与召回率、宏平均 F1、校准程度和置信区间
- **回归：** MAE、RMSE、偏差、残差图，以及按生物学或采集条件分组的误差
- **分割：** 各类 IoU/Dice、边界误差、目标级误差和失败案例
- **检测：** 指标定义、IoU 阈值范围、按目标大小分层的结果，以及精确率—召回率曲线

同时报告数据集组成和不确定性。总体得分高，可能掩盖了在某个品种、田块、相机或稀有类别上的失败。

## 11. 部署是一个独立的工程环节

在 notebook 里调用一次推理，不等于上线服务。部署前需要明确：

- 模型和预处理的版本；
- 输入格式、大小和内容的限制；
- 身份认证与权限控制；
- 请求频率和资源上限；
- 超时、批处理和并发策略；
- 隐私与数据保存策略；
- 监控、回滚和数据漂移检测；
- 用导出后的模型（而非训练时的模型）进行测试。

ONNX、`torch.export`、TorchScript、各类加速编译器和服务框架都有各自的版本限制，应在目标硬件上实测后再选择。没有上述保障的 Flask 接口，只能算本地演示，不能算生产部署。

## 常见问题排查

| 现象 | 优先检查 |
| --- | --- |
| 显存不足 | 输入尺寸、批大小、未释放的计算图、计算精度和无用张量 |
| 设备不一致 | 模型、输入、标签和新建的张量是否在同一设备上 |
| DataLoader 卡住 | 先设 `num_workers=0`，确认正常后再逐步增加 |
| 损失变成 NaN | 输入取值范围、标签、学习率、损失函数的前提、精度和梯度 |
| 结果不稳定 | 随机种子、数据顺序、数据增强、包版本、底层算子，以及划分是否泄漏 |
| 检查点加载失败 | 网络结构和类别映射是否一致；用指定设备加载可信的参数字典 |

## 建议的学习顺序

1. 张量、形状、数据类型和自动求导；
2. `Dataset`、`DataLoader` 和防止泄漏的数据划分；
3. 清晰的训练与验证循环；
4. 简单基线和明确的评价指标；
5. 预处理有版本记录的迁移学习；
6. 实验记录与消融实验；
7. 模型导出与部署验证。

## 官方资源

- [PyTorch 文档](https://pytorch.org/docs/)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [PyTorch 论坛](https://discuss.pytorch.org/)
- [PyTorch GitHub issues](https://github.com/pytorch/pytorch/issues)
