---
slug: root-quantify
title: "Root Quantify：用 Python 交互式预处理根系图像"
description: "一个交互式 OpenCV 工具：通过多边形 ROI 选择、光照校正、阈值分割和画笔修正，把原始根系扫描图和照片处理为干净的二值图像，可直接用于根系性状软件。"
authors: [liangchao]
tags: [python, image-analysis, plant-phenotyping]
layers: [DIG]
category: 植物表型
article_type: 研究工具
---

RhizoVision Explorer、WinRHIZO 等根系性状软件从二值图像计算根长、直径和根系构型，结果的好坏取决于这张二值图像。原始根系照片中有托盘、标签、不均匀光照、阴影和土壤颗粒，全自动阈值分割要么丢失细小侧根，要么保留杂质。

**Root Quantify** 是我为这一预处理环节编写的 OpenCV 桌面工具，将自动校正与有针对性的人工复核结合起来：用户圈定根系区域，工具校正不均匀光照并进行阈值分割，用户再用画笔修正剩余错误。输出为干净的二值根系图像，可直接导入根系性状软件。

<!-- truncate -->

## 处理流程

| 阶段 | 操作 | 结果 |
| --- | --- | --- |
| 文件夹扫描 | 查找 JPG、JPEG、PNG、BMP、TIF 和 TIFF 文件 | 待处理图像队列 |
| ROI 选择 | 用多边形圈定有效根系区域 | 掩膜后的裁剪图 |
| 预处理 | 估计背景、校正不均匀光照、阈值分割并反相 | 浅色背景上的深色根系 |
| 人工修正 | 用可调画笔补画或擦除像素 | 经复核的二值图像 |
| 导出 | 保存校正后的图像，并将原图移入归档文件夹 | 下次运行不会重复处理 |

校正后的图像在 RhizoVision Explorer、WinRHIZO、ImageJ 或实验室自有流程中进行测量。

## 安装

源代码见 [Root Quantify GitHub 仓库](https://github.com/smiler488/RootQuantify)。

```bash
git clone https://github.com/smiler488/RootQuantify.git
cd RootQuantify

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 下用以下命令激活环境：

```powershell
.venv\Scripts\activate
```

程序需要图形桌面环境；在无显示的服务器或 notebook 中，OpenCV 选择窗口无法显示。

运行前将 `RootImager.py` 中的 `folder_path` 设为图像所在目录。处理完成的原图会被移入 `processed_original`，因此请另行备份原始图像。

## 运行

```bash
python RootImager.py
```

程序使用两个窗口：一个始终显示原图，另一个用于 ROI 选择和修正。

### 键盘操作

| 按键 | 阶段 | 功能 |
| --- | --- | --- |
| `c` | ROI 选择 | 确认多边形（至少三个顶点） |
| `r` | ROI 选择 | 重置多边形 |
| `d` | 人工修正 | 补画深色根系像素 |
| `e` | 人工修正 | 擦除为浅色背景 |
| `+` / `-` | 人工修正 | 增大或减小画笔 |
| `u` | 人工修正 | 撤销上一笔 |
| `q` | 人工修正 | 完成当前图像 |

确认多边形后仔细检查自动阈值结果，只修正明显的分割错误；人工修改越多，可重复性越低，修改情况应记入实验记录。

## 输入与输出

校正后的图像写入 `output` 目录，文件名带 `processed-` 前缀；每张处理完成的原图移入 `processed_original`。

为保证可复现，与输出一并保存：

- 原始图像（单独存放的只读备份）；
- Root Quantify 的提交哈希；
- 脚本中的预处理参数；
- 操作者和修正日期；
- 对困难图像或剔除图像的说明。

## 质量控制清单

- [ ] 根系与背景的灰度差异明显；
- [ ] ROI 已排除标签、标尺、盆沿和无关物体；
- [ ] 阈值分割后细小侧根得以保留；
- [ ] 阴影未被误判为根系；
- [ ] 人工修正尽量少且有记录；
- [ ] 用于发表的测量，由第二人抽查样本；
- [ ] 下游测量已用已知尺寸物体或人工参考数据检验。

## 适用范围与局限

阈值分割对阴影、反光、基质和重叠根系敏感，这正是工具保留人工修正环节的原因；而人工修正会引入操作者差异，因此修改应尽量少并留有记录。二值输出不保留颜色和灰度信息。交互式设计面向实验规模图像集的精细处理，而非无人值守的高通量处理；性状测量本身在下游软件中完成。

实验室中另有浏览器版本：[根系图像预处理器](/app/root-processor)（[教程](/docs/tutorial-apps/root-preprocessor-tutorial)）。
