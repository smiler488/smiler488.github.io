---
slug: root-quantify
title: "Root Quantify：用 Python 交互式预处理根系图像"
description: "一份实用指南：用 Root Quantify 进行多边形 ROI 选择、背景校正、二值掩膜清理，并在下游根系分析前有序导出。"
authors: [liangchao]
tags: [python, image-analysis, plant-phenotyping]
category: 植物表型
article_type: 技术指南
---

**Root Quantify** 是一个用于准备根系图像的小型 OpenCV 桌面工具。它引导用户完成多边形选择、背景校正、二值化和手动清理，然后保存校正后的区域以便在另一个工具中分析。

准确描述其边界很重要：当前程序创建的是清理后的二值图像；它本身**不会**计算经过验证的根长、密度、直径或架构性状。

<!-- truncate -->

## 当前工具的功能

| 阶段 | 操作 | 结果 |
| --- | --- | --- |
| 文件夹扫描 | 查找 JPG、JPEG、PNG、BMP、TIF 和 TIFF 文件 | 一批源图像队列 |
| ROI 选择 | 记录有用根系区域周围的多边形顶点 | 一个掩膜裁剪 |
| 预处理 | 估计背景、减轻不均匀光照、阈值化并反转裁剪 | 浅色背景上的深色根系 |
| 手动校正 | 用可调画笔绘制或擦除像素 | 一张经审查的二值图像 |
| 导出 | 保存校正后的图像，并把原图移入归档文件夹 | 下次运行不会意外重复处理 |

这套工作流最适合在骨架化或测量之前使用，配合 RhizoVision Explorer、WinRHIZO、ImageJ 或经过验证的实验流程等软件。

## 安装

源码可在 [Root Quantify GitHub 仓库](https://github.com/smiler488/RootQuantify) 获取。

```bash
git clone https://github.com/smiler488/RootQuantify.git
cd RootQuantify

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在 Windows 上，用以下命令激活环境：

```powershell
.venv\Scripts\activate
```

该界面需要图形桌面。无头服务器或 notebook 会话在不额外配置显示的情况下无法显示 OpenCV 选择窗口。

:::caution 配置输入目录

当前 `RootImager.py` 修订版在脚本中包含一个 `folder_path` 值。运行前把它设为包含图像的目录。请保留该目录的备份，因为已完成的原图会被移入 `processed_original`。

:::

## 运行工作流

```bash
python RootImager.py
```

使用两个窗口：一个保持原图可见，另一个处理 ROI 选择和校正。

### 键盘控制

| 按键 | 场景 | 操作 |
| --- | --- | --- |
| `c` | ROI 选择 | 确认一个至少有三个顶点的多边形 |
| `r` | ROI 选择 | 重置多边形 |
| `d` | 手动校正 | 绘制深色根系像素 |
| `e` | 手动校正 | 擦除为浅色背景 |
| `+` / `-` | 手动校正 | 增大或减小画笔尺寸 |
| `u` | 手动校正 | 撤销上一次完成的笔触 |
| `q` | 手动校正 | 完成当前图像 |

确认多边形后，仔细检查自动阈值。只校正明显的分割错误；过度的人工编辑会降低可重复性，并应记录在实验日志中。

## 输入与输出

程序把校正后的图像写入 `output` 目录，文件名带 `processed-` 前缀。它把每张已完成的源图像移入 `processed_original`。

对于可复现的工作，连同输出一起保存以下内容：

- 未经修改的原图，存放在单独的只读备份中；
- Root Quantify 的提交哈希；
- 脚本中使用的预处理参数；
- 操作者身份和校正日期；
- 一段描述任何困难或被排除图像的说明。

不要把被移动的副本作为原始数据的唯一归档。

## 质量控制清单

- [ ] 根系和背景有可见的不同强度。
- [ ] ROI 排除了标签、标尺、盆边和无关物体。
- [ ] 细小的侧根在阈值化后被保留。
- [ ] 阴影没有被误认为根。
- [ ] 人工校正最少且有记录。
- [ ] 当测量将用于发表论文时，由第二位审查者抽查样本。
- [ ] 下游测量已对照已知物体或人工参考数据验证。

## 已知局限

- 基于阈值的分割对阴影、反光、基质和重叠根系敏感。
- 二值图像丢弃了原图中的颜色和强度信息。
- 人工校正引入操作者间变异。
- 移动源文件便于批处理，但需要有意的备份策略。
- 桌面交互不是为无人值守或高通量服务器处理设计的。
- 输出是预处理结果，不是生物学结论或定标后的表型表。

如需一种具有不同限制的浏览器端预处理工作流，请参见 [根系图像预处理器](/app/root-processor) 及其 [App Lab 教程](/docs/tutorial-apps/root-preprocessor-tutorial)。

*工作流审阅：2026 年 7 月。使用前请查阅仓库 README 和源码，因为界面可能变化。*
