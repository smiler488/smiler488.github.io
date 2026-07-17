---
title: NASA POWER 气象数据下载器
description: 选择位置和日期范围，获取 NASA POWER 农业气象数据，预览或下载整洁的 CSV 结果。
sidebar_label: 气象数据下载器
sidebar_position: 4
hide_title: true
keywords:
  - weather
  - climate
  - nasa
  - power
  - pcse
  - csv
app_route: /app/weather
app_icon: WX
app_category: Field planning
app_runtime: NASA POWER connection
app_tone: cyan
app_badges:
  - NASA POWER
  - Map selection
  - CSV export
---

## 功能简介

NASA POWER 气象数据下载器为一个坐标和日期范围请求固定的每日或每小时农业气象变量。你可以手动选择坐标，在地图上选择，通过地点搜索或使用浏览器定位，预览最多 100 行返回数据，并下载完整响应为 CSV。

:::info 原始 POWER 记录
当前应用导出 NASA POWER 返回的变量。它不会将其转换为 PCSE/WOFOST 字段、填充缺失值、计算 ET₀ 或生成图表。
:::

## 开始之前

- NASA POWER 和外部地图服务需要网络连接。
- 在纬度范围 `-90…90` 和经度范围 `-180…180` 内选择一个坐标。
- 准备包含起止日期。每日请求限制为 3,660 天，每小时请求限制为 366 天。
- 如果使用 **Get Current Location**（获取当前位置），请允许浏览器地理定位或仔细验证基于 IP 的近似回退坐标。

## 快速流程

1. 在 **Time Scale**（时间尺度）中选择 **Daily**（每日）或 **Hourly**（每小时）。对于每小时数据，在 **Time Standard**（时间标准）中选择 **LST** 或 **UTC**。
2. 输入纬度和经度，点击地图，使用 **Search**（搜索），或选择 **Get Current Location**。
3. 选择 **Start Date**（开始日期）和 **End Date**（结束日期）。
4. 选择 **Download NASA Weather Data**（下载 NASA 气象数据）并监控 **Status**（状态）。
5. 在 **Data preview**（数据预览）中检查前几行返回数据。
6. 选择 **Download CSV**（下载 CSV）以保存完整的返回表格。

## 控件与输出

| 控件或面板                             | 实际行为                                                                                  |
| -------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Time Scale**                         | 选择 NASA POWER 每日或每小时点数据。                                                      |
| **Time Standard**                      | 在每小时请求中发送 `LST` 或 `UTC`。                                                       |
| **Latitude / Longitude**               | 直接设置请求坐标。地图点击和定位工具更新这些字段。                                        |
| **Search place or address** / **Search** | 将查询发送到 Nominatim 并使用第一个结果。按 Enter 也会开始搜索。                          |
| **Get Current Location**               | 请求浏览器地理定位；在不支持、不安全、不可用或超时的情况下，可尝试基于 IP 的近似定位。    |
| **Download NASA Weather Data**         | 验证输入并请求固定参数集。后续请求会取消先前活跃的请求。                                  |
| **Data preview**                       | 最多显示前 100 条记录。                                                                   |
| **Download CSV**                       | 下载每条已解析记录，而非仅预览。                                                          |

每日参数：

```text
TOA_SW_DWN, ALLSKY_SFC_SW_DWN, T2M, T2M_MIN,
T2M_MAX, T2MDEW, WS2M, PRECTOTCORR
```

每小时参数：

```text
T2M, T2MDEW, RH2M, WS10M, U10M, V10M, PS, PRECTOT
```

## 工作原理

应用调用 NASA POWER 点 API，使用 community `AG`、所选坐标、包含日期范围和一个固定参数列表。每日请求使用每日端点。每小时请求使用每小时端点并包含所选时间标准。

返回的参数序列按其日期或小时键对齐并转换为行。预览渲染前 100 行，而 CSV 包含完整的已解析响应。值按返回原样导出；不进行统计清理、插值或单位转换。

## 数据、隐私与外部服务

此工作流连接到多个第三方：

| 服务          | 发送或请求的数据                                                                  |
| ------------- | --------------------------------------------------------------------------------- |
| NASA POWER    | 坐标、日期范围、时间尺度、适用时的时间标准和参数标识符。                          |
| OpenStreetMap | 可见区域的地图瓦片请求。                                                          |
| Nominatim     | 输入到地点搜索的文本。                                                            |
| ipapi.co      | 在精确定位无法使用后的基于 IP 的近似定位请求。                                    |
| unpkg         | 地图所需的 Leaflet JavaScript 和 CSS。                                            |

下载的记录保留在浏览器内存中，直到被替换或页面关闭。CSV 下载创建临时本地 URL，在清理期间撤销。

## 局限性

:::caution 模型输入质量

- NASA POWER 是网格化数据产品，非现场气象站测量。
- 可用性、单位、缺失值标记和质量因变量、时间产品、地点和时期而异；建模前请检查当前 POWER 元数据。
- 应用不移除哨兵值或验证农学一致性。
- 它提供每次请求一个位置，无月度聚合、批量处理、PCSE 转换、Excel 导出、图表或摘要统计。
- 地图、搜索、近似定位和数据检索取决于外部可用性、速率限制和浏览器网络策略。
  :::

## 排障

| 问题                              | 应检查什么                                                                                   |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| 地图不加载                        | 确认 Leaflet 和 OpenStreetMap 资源可访问。如果表单可用，仍可手动输入坐标。                   |
| 地点搜索无结果                    | 使用更具体的查询或手动输入坐标。                                                              |
| 定位不准确                        | IP 回退是近似的；在请求数据之前验证标记和坐标。                                              |
| 日期验证失败                      | 输入两个日期，保持开始日期在结束日期或之前，并在每日或每小时时长限制内。                      |
| NASA POWER 返回错误或无数据行     | 重新检查坐标和日期，缩短范围，确认 POWER 服务可用后重试。                                    |
| CSV 包含缺失或异常值              | 检查 POWER 元数据，在分析工作流中清理或转换导出数据。                                        |

[打开 NASA POWER 气象数据下载器](/app/weather)

[浏览全部应用](/app)
