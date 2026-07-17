# 任务：把 16 篇英文教程文档翻译成中文

> 这是交给翻译智能体的完整指令。把本文件全文复制给它即可。
> 配套校验脚本：`scripts/check-docs-i18n.mjs`

## 项目位置

```
/Users/dengliangchao/Documents/smiler488.github.io
```

## 你要做的事

读取 `docs/tutorial-apps/` 下的英文 markdown，翻译成中文，写入
`i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorial-apps/`。

**文件名必须完全相同，一个字符都不能改。** 系统靠文件名配对英文原文和中文译文，
改了网址就断、页面会 404。

例如：

- 读：`docs/tutorial-apps/weather-analyzer-tutorial.md`
- 写：`i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorial-apps/weather-analyzer-tutorial.md` ← 文件名一模一样

## 待翻译的 16 个文件

```
ai-data-visualizer-tutorial.md
ai-solver-tutorial.md
calibration-targets-tutorial.md
cco-mission-planner-tutorial.md
cloud-sticky-note-tutorial.md
custom-harvard-with-journal-abbr.md
image-quantifier-tutorial.md
irrigation-layout-designer-tutorial.md
journal-selector-tutorial.md
land-surveyor-tutorial.md
maze-game-tutorial.md
pptx-zh2en.md
root-preprocessor-tutorial.md
skill-creator-publish.md
stereo-camera-tutorial.md
weather-analyzer-tutorial.md
```

这 1 个**已经做完了，不要碰**：`sensor-app-tutorial.md`（作为标准样例）。

---

## 规则一：文件开头的 frontmatter（两个 `---` 之间的部分）

### 只翻这三个字段

| 字段 | 说明 |
| --- | --- |
| `title` | 文档标题 |
| `description` | 摘要，会显示在搜索结果里 |
| `sidebar_label` | 左侧边栏里显示的短名称 |

### 其余字段全部原样复制、一个字都不许改

包括但不限于：`sidebar_position`、`hide_title`、`keywords`、以及所有 `app_` 开头的字段
（`app_route`、`app_icon`、`app_category`、`app_runtime`、`app_tone`、`app_badges`）。

这些是机器读取的字段：`app_route` 是网址、`app_icon`/`app_tone` 是标识、`keywords` 保留英文即可。
**翻译它们会搞坏页面。** `app_category`、`app_runtime`、`app_badges` 的中文显示由网站代码
自动处理，你不用翻，保留英文原样即可。

### 正确示范

英文原文：

```yaml
---
title: NASA POWER Weather Downloader
description: Retrieve NASA POWER agrometeorological data and export clean CSV.
sidebar_label: Weather Downloader
sidebar_position: 4
hide_title: true
keywords:
  - weather
  - climate
app_route: /app/weather
app_icon: WX
app_category: Field planning
app_runtime: NASA POWER connection
app_tone: cyan
app_badges:
  - NASA POWER
  - Map selection
---
```

正确的中文版：

```yaml
---
title: NASA POWER 气象数据下载器
description: 获取 NASA POWER 农业气象数据，并导出整洁的 CSV。
sidebar_label: 气象数据下载器
sidebar_position: 4
hide_title: true
keywords:
  - weather
  - climate
app_route: /app/weather
app_icon: WX
app_category: Field planning
app_runtime: NASA POWER connection
app_tone: cyan
app_badges:
  - NASA POWER
  - Map selection
---
```

注意：只有 `title`、`description`、`sidebar_label` 变成了中文，**其余每一行都完全没动**。

### 错误示范（绝对不要这样）

```yaml
app_route: /应用/气象      ← 错！这是网址，永远不能翻
app_icon: 气象             ← 错！这是标识符
sidebar_position: 四       ← 错！这是排序数字，必须保持数字
```

---

## 规则二：正文里这些东西不要翻

1. **代码块内容不要翻。** 三个反引号包起来的内容原样保留，命令、代码、CSV 列名、参数名
   一个字符都不要动。唯一例外：代码块里的英文*注释*可以翻。

2. **界面按钮/控件名保留英文，可在括号里补中文。** 教程描述的是英文界面，用户看到的按钮是英文的。
   - 英文：`Tap **Capture Sample** and wait`
   - 中文：`点击 **Capture Sample**（采集样本），等待`
   - 表格里同理：`| **Enable sensors** | 请求运动/姿态权限…… |`（左列的控件名保留英文）

3. **链接的网址原样保留，只翻显示文字：**
   - 英文：`[Open Device Sensor Recorder](/app/sensor)`
   - 中文：`[打开设备传感器记录仪](/app/sensor)` ← `/app/sensor` 不动

4. **`:::info`、`:::caution`、`:::tip` 这类标记原样保留**，但它后面的标题文字和里面的内容要翻：

   ```
   :::caution 科研使用

   GPS 精度不等同于测量级精度……

   :::
   ```

   ↑ `:::caution` 和结尾的 `:::` 不动，其余翻译。

5. **表格的 `| --- | --- |` 分隔行不要动**，表头和单元格内容要翻（控件名按第 2 条保留英文）。

6. **专业术语保留英文**：CSV、GPS、NDVI、KML、UTM、CRS、NoData、EXIF、API、UAV 等。

---

## 规则三：翻译风格

**忠实翻译。** 贴近英文原意，只做必要的中文表达调整。

- 不要增加或删减内容
- 不要改变段落结构、标题层级、列表项数量、表格行数

---

## 做完后必须自查

在项目根目录运行：

```bash
node scripts/check-docs-i18n.mjs
```

**必须做到输出里没有任何 `✗` 才算完成。** 它会抓：`app_route` 等机器字段被改、
`sidebar_position` 被翻成中文、代码块反引号数量对不上、正文没有中文（忘了翻）。

如果报错，按提示修，再跑一遍，直到干净。然后跑构建确认没坏：

```bash
npm run build
```

---

## 参考样例

标准样例（格式照着它来）：

- 中文：`i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorial-apps/sensor-app-tutorial.md`
- 英文原文：`docs/tutorial-apps/sensor-app-tutorial.md`

开始前先对照读一遍，看清楚哪些翻了、哪些没翻。

---

## 最后提醒

不确定要不要翻的时候，**不要自己猜**。规则是：

> **能被机器读的 → 不翻**（frontmatter 里除 title/description/sidebar_label 外的一切、
> 代码、网址、界面按钮名）
>
> **给人看的 → 翻**（title、description、sidebar_label、正文段落）
