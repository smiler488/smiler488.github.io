---
title: "AI 数据可视化器"
description: "加载表格，创建确定性的本地图表，并可选地使用你选择的 AI 服务商来帮助解读数据规律。"
sidebar_label: "AI 数据可视化器"
sidebar_position: 10
hide_title: true
keywords:
  - "ai"
  - "chart"
  - "data"
  - "csv"
  - "xlsx"
  - "echarts"
app_route: "/app/ai-data-visualizer"
app_icon: "CSV"
app_category: "AI & research"
app_runtime: "Local charts · optional external AI"
app_tone: "blue"
app_badges:
  - "Local chart data"
  - "BYOK AI"
  - "CSV / XLSX / JSON"
---

## 功能简介

AI 数据可视化器在浏览器中加载表格，分析其列属性，并创建交互式 ECharts 可视化。默认的 **Local demo**（本地演示）直接根据上传的行数据构建确定性图表，无需网络请求。你也可以选择将一份精简的数据集摘要和分析目标发送给你配置的 AI 服务商。

:::info 本地优先工作流
原始文件在本地解析。当你需要直接基于行数据生成图表，而非由模型生成图表规格时，请使用 **Local demo** 或 **Apply field mapping**（应用字段映射）。
:::

## 开始之前

- 使用支持 File API、Canvas 和 JavaScript 的较新浏览器。
- 接受的文件格式为 CSV、TSV、XLSX、XLS 和 JSON，最大 20 MB。
- 一个有用的本地图表通常需要至少一个数值列。
- 实时 AI 分析需要网络访问、服务商/模型和你自己的 API 密钥。
- 直接浏览器调用仅在该服务商允许跨域请求（CORS）时才能工作。

## 快速流程

1. [打开 AI 数据可视化器](/app/ai-data-visualizer)。
2. 在 **Upload data**（上传数据）中，选择一个 CSV、TSV、Excel 或 JSON 文件。如果 Excel 工作簿包含多个工作表，请选择一个 **Excel sheet**（Excel 工作表）。
3. 在继续之前，检查预览和 **Dataset summary sent to AI**（发送给 AI 的数据集摘要）。
4. 输入 **Analysis goal**（分析目标）。
5. 可选设置 **X Field**（X 字段）、**Y Field**（Y 字段）、**Group**（分组）、**Aggregation**（聚合）、**Error Bars**（误差线）和 **Side-by-side multi charts**（并排多图表）。
6. 保持选择 **Local demo** 以获得无需网络的结果，或在 **Analysis model**（分析模型）下配置实时服务商。
7. 选择一个操作：
   - **Apply field mapping**（应用字段映射）立即根据所选字段重建本地图表。
   - **Generate visualization**（生成可视化）使用所选模型；使用 Local demo 时它也会生成一份确定性本地图表。
8. 检查 **AI Insights**（AI 洞察）、**Interactive chart**（交互式图表）和 **Raw AI response**（原始 AI 响应），然后在图表渲染完成后使用 **Download PNG**（下载 PNG）。

## 控件与输出

| 控件或输出                       | 用途                                                                       |
| -------------------------------- | -------------------------------------------------------------------------- |
| **Analysis goal**                | 描述你想要的对比、趋势、异常或图表。                                       |
| **X Field / Y Field**            | 选择类别列和数值列；**Auto**（自动）让本地启发式规则选择。                 |
| **Group**                        | 将数值拆分为多个系列。                                                     |
| **Aggregation**                  | 对映射字段计算均值、中位数、求和或计数。                                   |
| **Error Bars**                   | 当映射数据支持时，添加标准差或标准误的须线。                               |
| **Side-by-side multi charts**    | 将主图表与第二个本地视图组合。                                             |
| **Preview**                      | 显示已解析表格的精简样本。                                                 |
| **Dataset summary sent to AI**   | 显示实时服务商提示词中包含的确切精简文本。                                 |
| **AI Insights**                  | 显示返回的或本地生成的摘要和洞察列表。                                     |
| **Interactive chart**            | 渲染规范化后的 ECharts 选项。                                              |
| **Raw AI response**              | 显示实时模型 JSON 或本地模式图表载荷。                                     |

## 工作原理

浏览器最多读取 10,000 行表格数据，并保留一个较小的样本用于分析。发送给服务商的摘要包含文件元数据、列名、推断的数值列、类别示例、列属性以及最多前八行样本数据。其长度上限约为 8,000 个字符。

CSV 和 TSV 文件按分隔文本解析。Excel 工作簿按工作表解析。JSON 支持常见的表格结构，如对象数组、`columns` 加 `data` 结构，或 `headers` 加 `rows` 结构。

在 Local demo 模式下，图表值根据存储的行数据计算。在实时模式下，应用要求所选模型返回包含 `summary`、`insights` 和 `chart_option` 的严格 JSON，然后规范化该选项再渲染。当模型返回预期字段时，支持 Tukey 字母和高/低误差线。

如果实时请求失败且表格仍可在本地绘图，应用会显示本地回退图表。原始载荷可包含 `local-deterministic`、`offline-mapping`、`local-fallback` 或 `offline` 等模式。

## 数据、隐私与外部服务

完整源文件不会被本页面上传。本地解析、字段映射和图表渲染均在浏览器内存中进行。

当你选择实时服务商时，分析目标、字段映射和可见的精简数据集摘要会被发送到 **Analysis model** 中显示的 API 端点。该摘要可能包含样本值，因此在发送敏感或未发表的数据之前请仔细检查。

API 密钥仅保存在当前标签页的 React 状态中，在切换服务商、刷新或离开页面时清除，并发送到所显示的端点。静态网站无法像服务端代理那样保护浏览器输入的密钥。请使用受限的测试密钥；生产环境请使用你自己的已认证后端。

:::caution 校验模型生成的图表
外部模型可能省略行、编造数值或返回具有误导性的图表选项。在将结果用于研究或报告之前，请对照源表格校验坐标轴、数值、聚合和不确定性。
:::

## 局限性

- 超过 20 MB 的文件会被拒绝，仅存储前 10,000 行已解析数据。
- 精简的 AI 摘要并非完整数据集。
- 本地图表选择基于启发式规则，可能需要显式字段映射。
- 实时服务商可能因 CORS、账户策略、配额或模型访问权限而拒绝浏览器请求。
- 模型可能返回无效 JSON 或不受支持的 ECharts 选项。
- PNG 导出仅在 ECharts 成功渲染后才可用。

## 排障

- **文件被拒绝：** 确认文件扩展名并将其缩减到 20 MB 以下。
- **Excel 解析失败：** 用更简单的工作簿重试，或将所需工作表导出为 CSV。
- **没有可用的数值系列：** 清理数值单元格并选择显式的 X 和 Y 字段。
- **图表不是你期望的：** 设置字段映射、聚合和误差线，然后使用 **Apply field mapping**。
- **401、403、404 或 429：** 验证所选服务商、确切模型 ID、密钥权限、账单和配额。
- **CORS 或 Failed to fetch：** 使用 Local demo 或通过你已认证的后端路由请求。
- **图表渲染失败：** 简化目标或重新生成；返回的 ECharts 选项可能格式有误。
- **Download PNG 不可用：** 等待图表渲染并检查其上方是否有运行时错误。

[打开 AI 数据可视化器 →](/app/ai-data-visualizer)

[← 返回应用实验室](/app)
