---
slug: local-image-quantification-tutorial
title: "用 Python 做本地图像量化：一套可审计的实验工作流"
description: "一套紧凑的 OpenCV 工作流：分割分离的生物样本，导出像素和定标后的尺寸描述符，并记录科学使用前所需的验证。"
authors: [liangchao]
tags: [python, computer-vision, image-analysis, plant-phenotyping]
image: /img/blog-default.jpg
category: 植物表型
article_type: 技术指南
---

简单的阈值化和连通分量分析可以量化在均匀背景上拍摄的分离生物样本。这对原型、教学和受控筛选有用，但它不是通用的植物表型方法。

下面的工作流刻意要求一个明确的图像尺度。如果尺度定标失败，报告像素比虚构毫米更安全。

<!-- truncate -->

:::warning 实验性工作流

该示例并未针对每种相机、作物、样本类型、背景或光照条件进行验证。在把输出用于论文、育种决策或质控流程之前，请检查每个叠加图，并把一组有代表性的结果与独立人工测量进行比较。

:::

## 适用场景

本工作流假设：

- 样本在物理上是分离的；
- 背景大体均匀且在图像四角可见；
- 相机大致垂直于样本平面；
- 一个已知长度的参考物与样本在同一平面；
- 镜头畸变和透视可忽略或已校正；
- 目标性状可由二维轮廓近似。

它不适用于纠缠的根系、密集冠层、重叠叶片、不受控的田间场景，或三维曲率对测量至关重要的器官。

## 测量计划

在写代码之前，定义每个输出：

| 输出 | 操作性定义 |
| --- | --- |
| 面积 | 一个连通分量内的前景像素数，除以尺度的平方 |
| 周长 | 提取出的外轮廓长度 |
| 长和宽 | 最小面积旋转矩形的长边和短边 |
| 长宽比 | 旋转矩形的长除以宽 |
| 圆度 | `4π × 面积 / 周长²`；对轮廓噪声敏感 |
| 平均 RGB | 分量掩膜内源图像各通道的均值 |

这些是图像描述符。例如，最小矩形的长不自动等同于沿弯曲中脉的植物学叶长。

## 1. 拍摄受控图像

1. 使用漫射、稳定的光照和高对比度的哑光背景。
2. 保持相机距离、焦距、曝光、白平衡和朝向固定。
3. 把定标参考物放在样本平面内，而不是其上方或下方。
4. 避免触碰或重叠的样本。
5. 保留原始图像和元数据。
6. 在每次拍摄期间拍摄一张空背景和已知尺寸的验证物体。

如果颜色是一个结果，使用颜色靶和受控的颜色管理工作流。相机的 RGB 值依赖于设备和光照。

## 2. 显式计算尺度

用一张经审查的图像或定标工具，测量可见参考物的像素跨度：

```text
pixels_per_mm = reference_length_pixels / reference_length_mm
```

例如，一个 25 mm 的标记跨 310 像素，得到 `12.4 px/mm`。记录端点是如何选定的。当标记缺失或模糊时，不要回退到一个任意常数。

对于更高精度的工作，定标镜头畸变并从多个参考点估计一个平面变换，而不是依赖单一长度。

## 3. 安装最小环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python numpy pandas
python -m pip freeze > requirements-lock.txt
```

在 Windows 上，用 `.venv\Scripts\activate` 激活。

## 4. 运行一个紧凑的参考实现

把以下内容保存为 `quantify_samples.py`。它从图像四角估计背景，对颜色距离应用 Otsu 阈值，滤除小的连通分量，并导出 CSV 和叠加图。

```python
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    border = max(2, min(height, width) // 20)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    corners = [
        lab[:border, :border],
        lab[:border, -border:],
        lab[-border:, :border],
        lab[-border:, -border:],
    ]
    corner_pixels = np.vstack([block.reshape(-1, 3) for block in corners])
    background = np.median(corner_pixels, axis=0)

    distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
    distance_8bit = cv2.normalize(
        distance, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, mask = cv2.threshold(
        distance_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def quantify(image_path: Path, px_per_mm: float, min_area: int) -> None:
    if px_per_mm <= 0:
        raise ValueError("--px-per-mm must be a measured positive value")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    mask = foreground_mask(image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    overlay = image.copy()
    rows = []

    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < min_area:
            continue

        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter_px = float(cv2.arcLength(contour, True))
        (_, _), (side_a, side_b), angle = cv2.minAreaRect(contour)
        length_px, width_px = max(side_a, side_b), min(side_a, side_b)
        mean_b, mean_g, mean_r, _ = cv2.mean(image, mask=component)

        sample_id = f"S{len(rows) + 1}"
        rows.append(
            {
                "id": sample_id,
                "area_px": area_px,
                "perimeter_px": round(perimeter_px, 2),
                "length_mm": round(length_px / px_per_mm, 3),
                "width_mm": round(width_px / px_per_mm, 3),
                "area_mm2": round(area_px / px_per_mm**2, 3),
                "circularity": round(
                    4 * np.pi * area_px / max(perimeter_px**2, 1e-9), 4
                ),
                "angle_deg": round(float(angle), 2),
                "mean_r": round(mean_r, 1),
                "mean_g": round(mean_g, 1),
                "mean_b": round(mean_b, 1),
            }
        )

        box = np.intp(cv2.boxPoints(cv2.minAreaRect(contour)))
        cv2.drawContours(overlay, [contour], -1, (255, 120, 0), 2)
        cv2.drawContours(overlay, [box], -1, (0, 210, 255), 2)
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(
            overlay,
            sample_id,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    result = pd.DataFrame(rows)
    csv_path = image_path.with_name(f"{image_path.stem}_measurements.csv")
    overlay_path = image_path.with_name(f"{image_path.stem}_overlay.png")
    result.to_csv(csv_path, index=False)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Detected {len(result)} components")
    print(f"Wrote {csv_path} and {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--px-per-mm", type=float, required=True)
    parser.add_argument("--min-area", type=int, default=500)
    args = parser.parse_args()
    quantify(args.image, args.px_per_mm, args.min_area)
```

用测得的尺度运行它：

```bash
python quantify_samples.py samples.jpg --px-per-mm 12.4 --min-area 500
```

该脚本不会自动识别或排除定标物体。请在分析前裁掉它，或在检查叠加图并记录规则后仅移除其分量。

## 5. 接受数字之前先复核

对每批数据：

1. 把叠加图与原图在全分辨率下比较。
2. 确认生物样本的预期数量。
3. 剔除由阴影、标签、标记物或合并样本形成的分量。
4. 重复尺度测量并报告操作者间差异。
5. 人工测量一个盲测验证子集。
6. 以物理单位报告误差和置信区间。
7. 在声称泛化之前测试第二次拍摄。

保存剔除记录，而不是悄悄删除它们。

## 解读边界

- 下采样或图像压缩会改变轮廓和小结构。
- Otsu 阈值假设前景和背景分布可分。
- 旋转矩形会高估或错误表示强烈弯曲的样本。
- 周长和圆度对边界噪声高度敏感。
- 二维投影面积不是总叶面积或器官表面积。
- 平均 RGB 不是定标后的光谱测量或经验证的植被指数。
- 在光照、相机、背景、作物或生育期变化后，同样设置可能失效。

## 相关浏览器工具

[生物样本量化器](/app/image) 提供了一个托管图像分析服务的浏览器入口。在发送科研图像之前，请审阅其 [App Lab 教程](/docs/tutorial-apps/image-quantifier-tutorial)、服务状态、上传限制和外部处理隐私边界。

*参考实现审阅：2026 年 7 月。科学使用前请用你自己的采集流程验证。*
