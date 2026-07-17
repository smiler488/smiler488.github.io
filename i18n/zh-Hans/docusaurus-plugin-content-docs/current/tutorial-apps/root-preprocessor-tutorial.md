---
title: 根系图像预处理器
description: "使用多边形区域、背景清理、高通增强和精确手动修正来准备根系扫描图像。"
sidebar_label: 根系预处理器
sidebar_position: 7
hide_title: true
keywords: [root, scan, roi, segmentation, image]
app_route: /app/root-processor
app_icon: "ROI"
app_category: "Imaging & vision"
app_runtime: "Local image processing"
app_tone: green
app_badges: ["Canvas workflow", "ROI editing", "PNG export"]
---

## 功能简介

根系图像预处理器在浏览器中将根系扫描转换为可编辑的黑底白字掩码。你可以处理一小批图像，追踪多边形感兴趣区域，预览背景清理，在区域内应用高通增强，用画笔修正掩码，并导出 PNG。

:::info 本地工作流

图像解码、处理、编辑和导出在此浏览器标签页中完成。应用不会将根系扫描上传到服务器。

:::

## 开始之前

- 使用照明均匀且根系与背景之间对比清晰的 JPG 或 PNG 扫描。
- 一次最多可打开 6 张图像。每个文件不得超过 24 MB。
- 图像被缩放至最长边不超过 1800 像素，打开批次限制为 900 万解码像素。
- 在关闭或刷新标签页之前保存每个结果；图像和编辑历史仅存在于内存中。
- 将输出视为仍需视觉质量控制的预处理掩码。

## 快速流程

1. 选择 **Upload images**（上传图像）并选择一张或多张扫描。
2. 在 **Batch**（批次）中选择一张图像。
3. 在 **Polygon Mode**（多边形模式）下，在右侧画布上围绕根系区域点击，然后选择 **Close Polygon**（闭合多边形）。
4. 可选调整 **Background threshold**（背景阈值）和 **Noise kernel**（噪声核），然后选择 **Preview Background Cleanup**（预览背景清理）以查看单独的清理预览。
5. 调整 **Blur radius**（模糊半径）和 **ROI threshold**（ROI 阈值），然后选择 **Run ROI Processing**（运行 ROI 处理）。
6. 在 **Manual Brush**（手动画笔）中，选择 **Draw (black)**（绘制——黑色）以恢复根系或 **Erase (white)**（擦除——白色）以移除伪影。根据需要调整 **Brush size**（画笔大小）。
7. 使用 **Undo Brush Stroke**（撤销画笔笔画）进行近期修正，当掩码就绪时选择 **Download Processed PNG**（下载处理后的 PNG）。
8. 对剩余批次项目重复上述步骤，或使用 **Clear batch**（清除批次）从内存中移除所有打开的图像。

## 控件与输出

| 控件                                | 当前行为                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| Upload images                       | 在文件、批次和解码像素限制内打开 JPG/PNG 扫描。                                       |
| Batch                               | 在打开的图像之间切换，显示 `Pending` 或 `Processed`。                                 |
| Background threshold                | 设置背景清理预览的灰度截止值。                                                        |
| Noise kernel                        | 对背景清理预览应用形态学开运算。                                                      |
| Preview Background Cleanup          | 更新预览画布；它不替换最终 ROI 处理输入。                                              |
| Blur radius                         | 设置最终高通增强使用的局部模糊。                                                      |
| ROI threshold                       | 将归一化的高通响应转换为最终二值掩码。                                                 |
| Close / Undo Point / Reset Polygon  | 完成、缩短或清除多边形。多边形需要至少 3 个点，最多支持 60 个。                       |
| Run ROI Processing                  | 在闭合多边形内创建白底黑字根系掩码。                                                  |
| Polygon Mode / Manual Brush         | 在 ROI 定义和掩码编辑之间切换。手动模式需要已处理结果。                                |
| Draw / Erase / Brush size           | 在处理后的掩码上绘制黑色或白色笔画。                                                  |
| Undo Brush Stroke                   | 恢复最近的掩码快照；内存历史最多保留 4 个快照。                                       |
| Reset Processed Result              | 清除处理和画笔历史，但保留当前多边形，直到使用 **Reset Polygon**。                    |
| Download Processed PNG              | 将当前处理后的掩码导出为 `original-name-processed.png`。                               |

## 工作原理

浏览器解码每张图像并在将像素数据放入内存之前缩小超大尺寸。**Preview Background Cleanup** 将图像转换为灰度，进行阈值处理，并可选地应用形态学开运算；它是由 **Background threshold** 和 **Noise kernel** 控制的诊断预览。

最终的 **Run ROI Processing** 遵循单独的路径。它模糊原始图像，比较模糊和原始灰度强度，归一化高通响应，应用 **ROI threshold**，并仅在多边形内写入黑色检测。手动画笔然后直接编辑该二值结果。

这是一个受根系处理工作流启发的轻量级 Canvas 实现，不是 OpenCV 脚本的精确浏览器移植或经过验证的分割模型。

## 数据、隐私与外部服务

- 文件保留在当前浏览器标签页中，不会被应用传输。
- 不使用账户、云项目、自动保存或恢复服务。
- **Clear batch**、刷新、导航或关闭标签页会丢弃内存中的图像和编辑历史。
- PNG 下载使用 Canvas API 在本地创建。

## 局限性

:::caution 科研使用

生成的 PNG 是候选掩码，不是真实值的根系测量。在将其用于 ImageJ 或其他测量工作流之前，请检查细根、交叉点、扫描仪伪影和多边形边缘。

:::

- 未实现拖放；请使用 **Upload images**。
- 背景预览控件不影响最终 ROI 高通掩码。
- 大图像和大模糊半径在主线程上运行像素操作时可能阻塞浏览器。
- 自动缩放会减少高分辨率扫描中的精细细节。
- 撤销故意较浅以限制内存使用。
- 没有缩放工具、自动根系拓扑分析、批量导出或直接 WinRHIZO 集成。

## 排障

| 问题                                | 应检查什么                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| 文件被拒绝                          | 使用 JPG/PNG，保持在 24 MB 以下，并在 6 文件和 900 万像素批次限制内。                |
| **Run ROI Processing** 无反应       | 选择一张图像，添加至少 3 个多边形点，然后先选择 **Close Polygon**。                  |
| 细根消失                            | 降低 **ROI threshold** 或 **Blur radius**，重新运行处理，然后用 **Draw (black)** 恢复孤立细节。 |
| 仍有噪声                            | 提高 **ROI threshold**，收紧多边形，或用 **Erase (white)** 移除伪影。                |
| 多边形编辑被阻止                    | 选择 **Reset Processed Result**；如需新边界，也使用 **Reset Polygon**。              |
| 画笔编辑不可用                      | 在切换到 **Manual Brush** 之前先运行 ROI 处理。                                      |
| 浏览器变慢                          | 减小源尺寸，打开更少的图像，或使用更小的模糊半径。                                   |

[打开根系图像预处理器](/app/root-processor)

[返回应用实验室](/app)
