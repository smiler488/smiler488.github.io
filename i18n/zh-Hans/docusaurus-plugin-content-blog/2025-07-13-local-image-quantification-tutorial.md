---
slug: local-image-quantification-tutorial
title: "用 Python 从图像中量化种子、叶片等植物样本"
description: "一种本地运行的 OpenCV 方法：在均匀背景上分割彼此分离的植物样本，以定标后的物理单位测量尺寸、形状和颜色，并为每张图像输出标注叠加图。"
authors: [liangchao]
tags: [python, computer-vision, image-analysis, plant-phenotyping]
layers: [DIG]
image: /img/blog-default.jpg
category: 植物表型
article_type: 方法
---

种子、叶片、果实等器官平铺在均匀背景上拍照后，可以通过图像快速、客观地测量每个样本的面积、长、宽、形状和颜色，取代卡尺测量和目测打分。本文介绍一种紧凑的本地 Python 方法：按颜色距离把样本从背景中分割出来，逐个测量连通区域，用实测比例尺把结果换算为毫米，并为每张图像输出数据表和标注叠加图。全部计算在本地完成，图像不会离开电脑。

<!-- truncate -->

## 成像条件

该方法面向受控成像，要求：

- 样本彼此分开放置；
- 背景基本均匀，且在图像四角可见；
- 相机大致垂直于样本平面；
- 已知长度的参考物与样本处于同一平面；
- 镜头畸变和透视可忽略或已校正；
- 目标性状可由二维轮廓表示。

缠绕的根系、浓密冠层、相互重叠的叶片和田间场景需要专门的分割方法；三维曲面形态重要的器官则需要三维测量。

## 性状定义

每项输出都有明确的操作性定义：

| 输出 | 定义 |
| --- | --- |
| 面积 | 单个连通区域内的前景像素数，除以比例尺的平方 |
| 周长 | 外轮廓长度 |
| 长与宽 | 最小面积外接旋转矩形的长边与短边 |
| 长宽比 | 旋转矩形的长除以宽 |
| 圆度 | `4π × 面积 / 周长²`，对轮廓噪声敏感 |
| 平均 RGB | 区域掩膜内原图各通道的均值 |

以上均为基于图像的定义；对于明显弯曲的器官，最小外接矩形的长与沿中脉测得的植物学长度并不相同。

## 1. 受控拍摄

1. 使用稳定的漫射光和对比度高的哑光背景；
2. 固定相机距离、焦距、曝光、白平衡和朝向；
3. 定标参考物放在样本平面内，不高于或低于样本；
4. 样本互不接触、互不重叠；
5. 保留原始图像及元数据；
6. 每次拍摄时另拍一张空背景和若干已知尺寸的检验物。

若颜色是研究指标，需使用色卡和规范的色彩管理流程，因为相机 RGB 值随设备和光照而变。

## 2. 计算比例尺

在图像中测量参考物的像素长度：

```text
pixels_per_mm = reference_length_pixels / reference_length_mm
```

例如，25 mm 的标志在图像中跨 310 像素，则比例尺为 `12.4 px/mm`。脚本要求显式输入该值，使输出中的每个毫米值都能追溯到实测参考；端点的选取方式一并记录。精度要求更高时，先标定镜头畸变，再由多个参考点估计平面变换，而不是只依赖一段长度。

## 3. 安装环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python numpy pandas
python -m pip freeze > requirements-lock.txt
```

Windows 下用 `.venv\Scripts\activate` 激活。

## 4. 实现

将以下代码保存为 `quantify_samples.py`。程序由图像四角估计背景颜色，对 Lab 空间的颜色距离做 Otsu 阈值分割，滤除小连通区域，并导出 CSV 表和叠加图。

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

用实测比例尺运行：

```bash
python quantify_samples.py samples.jpg --px-per-mm 12.4 --min-area 500
```

定标物在分析前裁掉，或在检查叠加图后按记录的规则剔除其对应区域。

## 5. 质量控制

每批数据：

1. 以原始分辨率将叠加图与原图对照；
2. 核对样本数量是否与预期一致；
3. 剔除由阴影、标签、标志物或粘连样本形成的区域；
4. 重复测量比例尺，报告操作者间差异；
5. 对盲选的验证子集进行人工测量；
6. 以物理单位报告误差及置信区间；
7. 用另一次拍摄的数据检验方法的可推广性。

剔除的样本保留记录，而不是直接删除。

## 适用范围与局限

- 降采样和图像压缩会改变轮廓和细小结构；
- Otsu 阈值要求前景与背景的分布可以区分；
- 旋转矩形不能准确表示明显弯曲的样本；
- 周长和圆度对边界噪声高度敏感；
- 二维投影面积不等于叶片或器官的表面积；
- 平均 RGB 不是经过定标的光谱测量；
- 光照、相机、背景、作物或生育期改变后，需要重新检查参数。

## 相关浏览器工具

实验室中的[生物样本量化器](/app/image)通过托管服务在浏览器中提供同类分析（[教程](/docs/tutorial-apps/image-quantifier-tutorial)）；与本地脚本不同，它需要上传图像进行处理。
