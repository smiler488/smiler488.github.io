---
title: CCO 航线构建器
description: 将 KML 田块边界转换为可预览的 DJI 兼容 CCO 航线包，支持可配置的飞行和相机设置。
sidebar_label: CCO 任务规划器
sidebar_position: 5
hide_title: true
keywords:
  - drone
  - uav
  - cco
  - kml
  - kmz
  - wayline
app_route: /app/cco
app_icon: UAV
app_category: Field planning
app_runtime: Local route generation
app_tone: orange
app_badges:
  - KML / KMZ
  - Route preview
  - DJI waylines
---

## 功能简介

CCO 航线构建器读取 KML 多边形，在其周围放置旋转的圆形航线中心点网格，构建蛇形排序的相机朝向航点序列，并导出 KML、WPML 以及在 JSZip 可用时导出 KMZ 文件。它还可以从现有 DJI KMZ 中读取无人机和载荷枚举值。

:::caution 飞行规划预览
在操作飞行器之前，请在 DJI 软件中验证航线几何、高度解释、设备枚举值、载荷动作、障碍物、空域和当地飞行法规。
:::

## 开始之前

- 准备一个 `.kml` 文件，其第一个可用几何为目标多边形。KML 输入限制为 5 MB 和 10,000 个顶点。
- 使用适合 KML 的地理经度/纬度坐标。
- 获取正确的 DJI 无人机和载荷枚举值，或准备一个不超过 25 MB 的 DJI `.kmz` 以便从中读取。
- 保持航线足够小以适应浏览器和预期飞行器工作流。应用将网格中心点限制为 15,000 个，预估航点限制为 120,000 个。
- 需要网络访问以从 CDN 加载 JSZip，用于 KMZ 导入和导出。

## 快速流程

1. 在 **Target Area**（目标区域）下，选择一个多边形 KML 文件。
2. 配置 **Coverage Parameters**（覆盖参数），特别是 **Circle radius (m)**（圆半径，米）、**Pts/circle**（每圆点数）、**Overlap (0~0.9)**（重叠度）和 **Grid bearing (°)**（网格方位角）。
3. 配置 **Flight & Camera**（飞行与相机）并验证 **Drone & Payload (Optional)**（无人机与载荷——可选）值。可选上传 DJI KMZ 并选择 **Parse Drone & Payload**（解析无人机与载荷）。
4. 选择 **Preview**（预览），在 **Live Preview**（实时预览）中检查田块边界、中心点和航线。
5. 调整参数并再次选择 **Preview**，直到航线适合验证。
6. 选择 **Generate files**（生成文件），然后下载 `template.kml`、`waylines.wpml`、`cco_full.kmz` 或任何生成的拆分部分。

## 控件与输出

| 分组                                               | 输入或输出                                                                              |
| -------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Target Area**                                    | 一个 `.kml` 上传，最大 5 MB。解析器使用第一个匹配的多边形坐标元素。                     |
| **Circle radius / Pts/circle**                     | 设置每个采样环的半径和航点数（`3–360`）。                                               |
| **Overlap (0~0.9)**                                | 控制自动中心点间距，而非摄影正向或侧向重叠。                                            |
| **Center step**                                    | 使用输入的米间距；`0` 选择自动公式。                                                    |
| **Padding / Grid bearing / Start bearing**         | 扩展工作边界，旋转中心点网格并旋转每个环上的第一个点。                                  |
| **Center mode**                                    | 使用多边形质心或边界框中心作为局部网格原点。                                            |
| **Clip inside**                                    | 仅保留落在多边形内的生成环航点。                                                        |
| **Prune outside centers**                          | 裁剪关闭时，如果某个中心点的所有采样环点都不在多边形内，则移除该中心点。                |
| **Altitude / Speed / Gimbal pitch / File suffix**  | 将固定任务值和每航点值写入生成的 XML。                                                  |
| **Drone & Payload**                                | 接受数字 DJI 枚举值；应用不会从中识别模型名称。                                         |
| **Max points / part**                              | 当航线超过此值时拆分输出；`0` 禁用拆分。                                                 |
| **Live Preview**                                   | 显示多边形、保留的中心点、航点路径、起点和终点标记。                                    |

## 工作原理

1. 解析器读取第一个匹配的 KML 坐标元素，移除重复的闭合顶点，并验证基本数值坐标和限制。
2. 在所选多边形中心附近构建局部米/度近似。
3. 中心点在填充边界范围内生成，按 **Grid bearing** 旋转，并以交替行排序。
4. 当 **Center step** 设为 `0` 时，间距为：

   ```text
   max(2 × circle radius × (1 − overlap), 1 metre)
   ```

5. 每个保留的中心点生成一个航点环。相邻环反向，下一个环旋转以在前一个端点附近开始。航点朝向环中心。
6. **Generate files** 在每个航点写入拍照动作，并创建 `template.kml`、`waylines.wpml` 以及在 JSZip 可用时创建 `wpmz/` KMZ 包。
7. 如果航线超过 **Max points / part**，航点列表被划分为顺序 KML/WPML 部分，并提供可选 KMZ 下载。

DJI KMZ 导入按顺序搜索 `waylines.wpml`、`wpmz/waylines.wpml`、`template.kml`、`wpmz/template.kml`、`doc.kml`，然后是任何 WPML 或 KML 文件。缺失的枚举字段回退到应用默认值，仍须验证。

## 数据、隐私与外部服务

KML/KMZ 读取、几何生成、预览和文件构建在浏览器中运行；上传的航线文件不会发送到航线生成服务器。临时下载 URL 在被替换或页面清理时撤销。

JSZip 从 jsDelivr 加载。如果不可用，纯 KML 和 WPML 生成仍可工作，但 KMZ 解析和 KMZ 打包不可用。

## 局限性

- 航线是确定性局部网格启发式，不是图像重叠、GSD、电池、时间、地形或风的优化器。
- 应用不配置传感器尺寸、焦距、ISO、快门速度或光圈。
- 它不检查障碍物、高程模型、地理围栏、空域、起飞位置、无线电链路、电池更换或返航路径。
- 局部经度/纬度近似不适用于非常大的区域、高纬度或跨越日期变更线的多边形。
- 不支持 MultiPolygon 几何、孔洞和完整 KML 模式语义。
- "DJI 兼容"输出仍取决于目标软件、飞行器、固件、枚举值和载荷配置。
- 拆分创建顺序文件，但不在各部分之间添加安全过渡、起飞或降落逻辑。

## 排障

| 问题                                  | 应检查什么                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| KML 被拒绝                            | 确认文件是有效 XML，低于 5 MB，且包含至少三个数值顶点的多边形坐标元素。                    |
| 预览报告中心点或点太多                | 增大 **Center step**，减小 **Padding**，减少 **Pts/circle** 或使用更小的目标区域。         |
| 预览没有航点                          | 重新检查 **Clip inside**、**Prune outside centers**、半径、间距和多边形几何。              |
| **Generate files** 要求预览           | 在最新的 KML 或覆盖参数更改后选择 **Preview**。                                            |
| KMZ 解析失败                          | 确认文件低于 25 MB，包含 KML/WPML，且 JSZip CDN 已加载。如有需要，手动输入枚举值。         |
| 生成的文件被 DJI 软件拒绝             | 在目标 DJI 版本中验证设备枚举值、高度、速度、载荷位置、WPML 预期和航线大小。               |

[打开 CCO 航线构建器](/app/cco)

[浏览全部应用](/app)
