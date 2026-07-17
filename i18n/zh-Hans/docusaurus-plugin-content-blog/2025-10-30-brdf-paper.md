---
slug: brdf-paper
title: 从表型性状预测叶片 BRDF
authors: [liangchao]
category: 植物表型
article_type: 研究项目
tags: [plant-phenotyping, remote-sensing, machine-learning, crop-modeling]
image: /img/brdf_cover.jpg
description: 一个经过同行评审的框架，结合方向光谱测量、BRDF 拟合、表型性状、集成学习和冠层光线追踪，覆盖四个物种。
---
import AltmetricBadge from '@site/src/components/AltmetricBadge';

## 概述

![方向光谱测量与 BRDF 预测工作流](/img/brdf_cover.jpg)

叶片表面并不均匀反射光。其解剖结构、色素和微观粗糙度会改变辐射在冠层中的散射方式，然而许多冠层模型使用简化的光学输入。

本研究结合自制的**方向光谱检测仪（DSDI）**、Cook–Torrance **双向反射分布函数（BRDF）** 拟合、表型测量和集成学习。目标是从更易测量的性状估计叶片光学参数，然后考察这些参数如何影响模拟的冠层光分布。

<!-- truncate -->

<AltmetricBadge doi="10.1016/j.plaphe.2025.100135" badgeType="donut" className="brdfAltmetric" />

## 速览

- **植物材料：** 玉米、水稻、棉花和杨树的叶片，取自冠层上部和下部位置。
- **方向光谱：** 400–1000 nm，用 DSDI 在宽角度范围内测量。
- **BRDF 参数：** 粗糙度 $\sigma(\lambda)$、漫反射系数 $k(\lambda)$ 和折射率 $n(\lambda)$。
- **预测模型：** 由支持向量、随机森林和梯度提升回归器构成的堆叠集成。
- **报告性能：** BRDF 拟合 $R^2 > 0.95$；集成预测 $R^2 = 0.83$–$0.99$，取决于参数。

## 测量与建模工作流

### 1. 测量方向反射率

DSDI 使用氙灯光源、光纤光谱仪和机械控制的照明与观测角度。在叶片测量前，用朗伯白板定标反射率。

叶片的上表面和下表面均被测量。这很重要，因为两个表面在表皮结构和光学响应上不同。

### 2. 拟合 BRDF 模型

Cook–Torrance 公式用三个随波长变化的参数表示漫反射和镜面反射：

| 参数 | 物理含义 | 相关叶片属性 |
| --- | --- | --- |
| $\sigma(\lambda)$ | 微 facet 粗糙度 | 表皮纹理和表面不规则性 |
| $k(\lambda)$ | 漫反射系数 | 内部散射和对反射率的漫反射贡献 |
| $n(\lambda)$ | 折射率 | 折射和界面反射，受组织组成影响 |

使用自适应网格搜索和最小二乘优化，把这些参数拟合到测得的方向光谱上。

### 3. 从性状预测光学参数

输入变量包括叶厚、比叶重、色素测量、显微镜导出的表面粗糙度和波长。堆叠模型结合了：

- 支持向量回归（SVR）
- 随机森林回归（RFR）
- 梯度提升回归树（GBRT）
- 以线性回归作为元学习器

所得模型在研究域内提供了一个直接的、数据驱动的链接，把测得的表型性状与 BRDF 参数联系起来。

### 4. 检验冠层尺度的后果

预测的 BRDF 参数被引入基于 **fastTracer** 的水稻冠层光线追踪工作流。模拟表明，改变粗糙度、漫反射或折射行为可以改变冠层内部光的垂直和角度分布。

## 结果支持什么

本研究支持三个实际结论：

1. 方向叶片反射率可以用基于物理的 BRDF 模型准确表示。
2. 结构和生化叶片性状包含预测 BRDF 参数的有用信息。
3. 叶片光学多样性可以在实质上改变模拟的冠层光场，不应总被当作均匀处理。

这些结果提供了一条把叶片尺度表型与辐射传输和冠层光合模型联系起来的途径。

## 范围与局限

该模型从覆盖四个物种、两个冠层位置和两个叶面的 **270 条数据**开发而来。因此它是一个研究模型，不是适用于每种作物、基因型、环境或胁迫处理的通用估计器。

- 在测得的性状和波长范围之外的预测需要新的验证。
- 光线追踪结果展示了模拟光分布的变化；它们本身并不证明田间的产量提升。
- 在处理新物种或需要高精度光学参数时，直接光学测量仍然重要。
- 未来数据集应覆盖更多基因型、环境、发育阶段和水分状况。

## 代码与数据可用性

- [BRDF 拟合脚本与粗糙度计算器](https://github.com/PlantSystemsBiology/brdf)
- [fastTracer 冠层光线追踪软件](https://github.com/PlantSystemsBiology/fastTracerPublic)
- 如已发表论文所述，研究数据可向通讯作者合理请求获取。

## 引用

Deng, L., Yu, L. X., Mao, L., Wang, Y., Guo, X., Wang, M., Zhang, Y., Song, Q., & Zhu, X.-G. (2025). **Leaf bidirectional reflectance distribution function (BRDF) prediction with phenotypic traits in four species: Development of a novel measuring and analyzing framework.** *Plant Phenomics, 7*(4), 100135. [https://doi.org/10.1016/j.plaphe.2025.100135](https://doi.org/10.1016/j.plaphe.2025.100135)
