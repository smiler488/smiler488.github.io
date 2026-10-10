---
slug: brdf-paper
title: 基于表型性状预测叶片 BRDF
authors: [liangchao]
category: 植物表型
article_type: 研究项目
tags: [plant-phenotyping, remote-sensing, machine-learning, crop-modeling]
layers: [UND]
image: /img/brdf_cover.jpg
description: 一套由表型性状预测四个物种叶片 BRDF 参数的测量与分析框架，结合方向光谱测量、Cook–Torrance 模型拟合、集成学习和冠层光线追踪（Plant Phenomics，2025）。
---
import AltmetricBadge from '@site/src/components/AltmetricBadge';

## 概述

![方向光谱测量与 BRDF 预测流程](/img/brdf_cover.jpg)

叶片并不是向各个方向均匀反射光。叶片的解剖结构、色素和表面微观粗糙度决定了光的散射方式，进而决定了光在冠层内的分布；但多数冠层模型仍把叶片当作朗伯反射体处理。而逐片测量叶片的方向反射又难以实现。

本研究建立了一套由易测表型性状预测叶片光学特性的框架：结合自主研制的**方向光谱检测仪（DSDI）**、Cook–Torrance **双向反射分布函数（BRDF）**模型拟合、表型测量和集成学习，并通过冠层光线追踪量化预测的光学特性对冠层光分布的影响。

<!-- truncate -->

<AltmetricBadge doi="10.1016/j.plaphe.2025.100135" badgeType="donut" className="brdfAltmetric" />

## 要点

- **材料：** 玉米、水稻、棉花和杨树叶片，分别取自冠层上部和下部。
- **方向光谱：** 用 DSDI 在大角度范围内测量 400–1000 nm 光谱。
- **BRDF 参数：** 粗糙度 $\sigma(\lambda)$、漫反射系数 $k(\lambda)$ 和折射率 $n(\lambda)$。
- **预测模型：** 由支持向量回归、随机森林和梯度提升回归构成的堆叠集成模型。
- **性能：** BRDF 拟合 $R^2 > 0.95$；集成模型预测 $R^2 = 0.83$–$0.99$（因参数而异）。

## 测量与建模

### 1. 方向反射测量

DSDI 由氙灯光源、光纤光谱仪以及机械控制的入射角和观测角组成。叶片测量前用朗伯白板标定反射率。叶片近轴面和远轴面均进行测量，因为两面的表皮结构和光学响应不同。

### 2. BRDF 模型拟合

Cook–Torrance 模型用三个随波长变化的参数描述漫反射和镜面反射：

| 参数 | 物理含义 | 相关叶片特性 |
| --- | --- | --- |
| $\sigma(\lambda)$ | 微面元粗糙度 | 表皮纹理与表面起伏 |
| $k(\lambda)$ | 漫反射系数 | 叶内散射及其对反射的漫反射贡献 |
| $n(\lambda)$ | 折射率 | 折射与界面反射，受组织成分影响 |

采用自适应网格搜索结合最小二乘优化，将三个参数拟合到实测方向光谱。

### 3. 由性状预测光学参数

输入变量包括叶片厚度、比叶重、色素含量、显微图像得到的表面粗糙度以及波长。堆叠模型由以下部分组成：

- 支持向量回归（SVR）
- 随机森林回归（RFR）
- 梯度提升回归树（GBRT）
- 以线性回归作为元学习器

该模型把实测表型性状与 BRDF 参数直接联系起来，无需方向光谱测量即可估计叶片光学特性。

### 4. 冠层尺度效应

将预测的 BRDF 参数引入基于 **fastTracer** 的水稻冠层光线追踪。模拟结果表明，粗糙度、漫反射和折射特性的变化会改变冠层内光的垂直分布和角度分布。

## 主要结论

1. 基于物理的 BRDF 模型能够准确描述叶片的方向反射；
2. 叶片的结构与生化性状包含预测 BRDF 参数所需的信息；
3. 叶片光学特性的差异会显著改变冠层内的模拟光场，因此应在模型中显式表示叶片光学特性，而不是假设其均一。

这些结果把叶片尺度的表型测量与辐射传输模型、冠层光合模型联系了起来。

## 适用范围与局限

模型基于覆盖四个物种、两个冠层位置和叶片两面的 **270 条数据**建立，适用于已测的性状范围和波长范围（400–1000 nm）；用于新物种和新条件时，需要补充测量以扩展模型。冠层模拟量化了光分布的变化，与田间产量的联系是下一步工作。将数据扩展到更多基因型、环境、发育阶段和水分状况，是本研究的后续方向。

## 代码与数据

- [BRDF 拟合脚本与粗糙度计算器](https://github.com/PlantSystemsBiology/brdf)
- [fastTracer 冠层光线追踪软件](https://github.com/PlantSystemsBiology/fastTracerPublic)
- 如论文所述，研究数据可向通讯作者合理索取。

## 引用

Deng, L., Yu, L. X., Mao, L., Wang, Y., Guo, X., Wang, M., Zhang, Y., Song, Q., & Zhu, X.-G. (2025). **Leaf bidirectional reflectance distribution function (BRDF) prediction with phenotypic traits in four species: Development of a novel measuring and analyzing framework.** *Plant Phenomics, 7*(4), 100135. [https://doi.org/10.1016/j.plaphe.2025.100135](https://doi.org/10.1016/j.plaphe.2025.100135)
