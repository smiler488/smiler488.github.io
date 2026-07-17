---
slug: pytorch-ml-dl-tutorial
title: 面向科研的 PyTorch——一份注重版本的学习路线图
description: 一条简洁、聚焦可复现性的路线：从张量与训练循环，到迁移学习、评估、安全检查点和部署边界。
authors: [liangchao]
category: 人工智能与机器学习
article_type: 技术指南
tags: [machine-learning, artificial-intelligence, python, computer-vision]
image: /img/blog-default.jpg
---

## 项目概述

本文是一份学习路线图，不是一份可复制即运行的生产框架。它保留一个小巧可运行的示例，然后解释让科研模型可审计的那些决策：设备处理、随机种子、数据划分、指标、检查点、版本记录和验证。

- **受众：** 开始可复现机器学习实验的 Python 用户
- **API 范围：** 近期的 PyTorch 2.x 和 torchvision 版本；请始终查阅已安装版本的文档
- **验证边界：** 示例刻意保持小巧且可在语法层面检查；不暗含任何基准、云价格或硬件性能声明

<!-- truncate -->

## 1. 从官方选择器安装

PyTorch 包依赖操作系统和加速器运行时。请使用 [PyTorch — Start Locally](https://pytorch.org/get-started/locally/) 上的当前选项，而不是照抄旧教程里的某个 CUDA URL。

仅 CPU 的 pip 安装适用于学习和持续集成：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

在 Windows 上，用适合 PowerShell 或命令提示符的命令激活环境。对于 CUDA、ROCm、XPU 或其他后端，请使用官方选择器并记录确切的命令。

记录环境：

```bash
python --version
python -m pip freeze > requirements-lock.txt
```

对于需要长期维护的项目，优先使用依赖文件和有意的锁定/更新流程，而非未经审查的快照。

## 2. 检查运行时并选择设备

不要在项目中散布无条件的 `.cuda()` 调用。选择一次设备，并把模型和张量都移到它上面。

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

把设备、加速器型号、驱动/运行时、包版本和精度模式随实验结果一起记录。

## 3. 在创建模型前设置可复现性控制

随机种子必须在初始化模型或打乱数据之前设置。

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

随机种子能提高可重复性，但不能保证跨 PyTorch 版本、设备、内核或分布式配置得到完全一致的结果。请报告容差并重做重要实验。

## 4. 训练一个最小模型

XOR 示例演示了张量、一个模块、logits、损失、优化器和推理，且不需要大数据集。

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

使用 `BCEWithLogitsLoss` 配合原始 logits，而不是在 `BCELoss` 之前再加一个 sigmoid 层。不要公布精确的预期概率：初始化、包版本和数值内核都可能改变它们。

## 5. 组织科研数据集

让数据集划分独立于在数据上拟合的预处理。

1. 定义观测单元；
2. 按一个能防止泄漏的单元划分，例如植株、小区、田块、日期或受试者；
3. 仅在训练划分上拟合归一化和特征变换；
4. 冻结验证划分用于模型选择；
5. 仅在最终评估时才触碰测试划分。

对于图像表型，按图像随机划分可能把同一植株的近乎重复视图泄漏到训练集和验证集中。分组感知划分通常更有说服力。

自定义数据集应返回样本和目标，且不悄悄改变全局状态：

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

构造 `rows` 时验证路径和标签，并记录缺失或损坏样本的处理方式。

## 6. 使用透明的训练循环

一个 epoch 应有清晰的契约：消费一个加载器、更新模型，并返回按样本加权的指标。

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
- 绝不更新模型参数；
- 在所有样本上汇总指标；
- 需要做误差分析时保存预测和标识符。

避免根据测试集表现来选择模型。

## 7. 仅在受支持时才加入混合精度

自动混合精度能提高 CUDA 吞吐量，但它是优化，不是正确性要求。先建立一个全精度基线。

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

旧版的 `torch.cuda.amp.autocast` 和 `torch.cuda.amp.GradScaler` 命名空间在当前文档中已被弃用。更改精度时，监控损失、梯度和指标是否出现 NaN 或溢出。

## 8. 从一个受维护的基线开始做计算机视觉

torchvision 的权重枚举把预训练参数与有文档的预处理绑定在一起：

```python
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 4)
```

记录权重枚举、输入分辨率、变换、类别映射和微调策略。不要把单个输出头称为一个完整的 YOLO 检测器：目标检测还需要目标编码、损失、解码、非极大值抑制、评估和任务专属训练。

对于植物图像，把学习到的模型与简单基线比较，并在重要的领域上评估——品种、生育期、传感器、田块、日期和光照。

## 9. 安全地保存状态

保存状态字典和元数据，而不是序列化一个任意的活 Python 对象：

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

只加载受信任的文件，并映射到已知设备：

```python
checkpoint = torch.load(
    "checkpoints/model.pt",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state_dict"])
```

`torch.load` 在 `weights_only=True` 时使用受限加载，但项目仍应把外部模型产物视为不受信任的输入，并核验其来源和完整性。

## 10. 评估不止一个数字

在看最终结果之前先选好指标。

- **分类：** 混淆矩阵、每类精确率/召回率、宏 F1、校准和不确定区间
- **回归：** MAE、RMSE、偏差、残差图，以及按生物学或采集子组划分的误差
- **分割：** 每类 IoU/Dice、边界误差、目标级误差和失败案例
- **检测：** 指标定义、IoU 范围、目标尺寸分层和精确率-召回率曲线

报告数据集构成和不确定性。一个高的汇总分数可能掩盖在某个品种、田块、相机或稀有类别上的失败。

## 11. 把部署当作一个独立的工程阶段

notebook 里的一次推理调用不是生产服务。部署之前，定义：

- 模型和预处理版本；
- 输入模式、尺寸和内容限制；
- 认证和授权；
- 请求速率和资源限制；
- 超时、批处理和并发行为；
- 隐私和数据留存策略；
- 可观测性、回滚和漂移监控；
- 使用导出产物而非训练模型所做的测试。

ONNX、`torch.export`、TorchScript、加速器编译器和服务框架都有版本相关的约束。只有在目标硬件上测量之后才选定一种。一个没有这些控制的 Flask 端点应被描述为本地演示，而非生产部署。

## 故障排查清单

| 症状        | 首要检查                                                                                      |
| ----------- | --------------------------------------------------------------------------------------------- |
| 显存不足    | 输入尺寸、批大小、保留的计算图、精度和未使用的张量                                            |
| 设备不匹配  | 模型、输入、目标和新创建的张量使用同一设备                                                    |
| DataLoader 卡住 | 先用 `num_workers=0`，再在测试平台的同时增加                                                  |
| 损失为 NaN  | 输入范围、标签、学习率、损失假设、精度和梯度                                                  |
| 结果变化    | 随机种子、数据顺序、数据增强、包版本、内核和划分泄漏                                          |
| 检查点加载失败 | 架构和类别映射匹配；用显式设备加载受信任的状态字典                                            |

## 建议的学习顺序

1. 张量、形状、dtype 和 autograd；
2. `Dataset`、`DataLoader` 和防泄漏划分；
3. 一个透明的训练与验证循环；
4. 一个简单基线和有文档的指标；
5. 带版本化预处理的迁移学习；
6. 实验追踪和消融研究；
7. 导出和部署验证。

## 官方资源

- [PyTorch 文档](https://pytorch.org/docs/)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [PyTorch 论坛](https://discuss.pytorch.org/)
- [PyTorch GitHub issues](https://github.com/pytorch/pytorch/issues)
