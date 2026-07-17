# 任务：把 17 篇英文博客翻译成中文

> 这是交给翻译智能体的完整指令。把本文件全文复制给它即可。
> 配套校验脚本：`scripts/check-blog-i18n.mjs`

## 项目位置

```
/Users/dengliangchao/Documents/smiler488.github.io
```

## 你要做的事

读取 `blog/` 下的英文 markdown，翻译成中文，写入 `i18n/zh-Hans/docusaurus-plugin-content-blog/`。

**文件名必须完全相同，一个字符都不能改。** 系统靠文件名配对英文原文和中文译文，改了就失效。

例如：

- 读：`blog/2024-05-03-root-quantify.md`
- 写：`i18n/zh-Hans/docusaurus-plugin-content-blog/2024-05-03-root-quantify.md` ← 文件名一模一样

## 待翻译的 17 个文件

```
2021-01-08-personal-website-docusaurus-github-pages.md
2021-02-15-macos-shortcuts.md
2021-04-24-gitHub-beginner-guide.md
2022-02-26-workflow-vscode-miniconda-git.md
2022-04-28-growth-chamber-cotton-3d.md
2022-09-20-pytorch-ml-dl-tutorial.md
2023-08-20-canopy-photosynthesis-modeling.md
2024-05-03-root-quantify.md
2024-07-07-uav-3d-crop-phenotyping.md
2025-02-20-hunyuan3d-plant-reconstruction-guide.md
2025-06-17-academic-paper-publication-guide.md
2025-07-13-local-image-quantification-tutorial.md
2025-08-20-local-ai-agent-deployment.md
2025-10-30-brdf-paper.md
2025-11-07-mctp-unified-phenotyping-platform.md
2025-12-13-botanical-extract-ai-pro.md
2026-01-06-phenohub-wechat-miniapp.md
```

这 2 个**已经做完了，不要碰**：

- `2023-04-22-dji-p4m-webodm-qgis-tutorial.md`
- `2024-01-11-local-llm-training-guide.md`

---

## 规则一：文件开头的 frontmatter（两个 `---` 之间的部分）

### 必须原样复制、一个字都不许翻的字段

| 字段 | 原因 |
| --- | --- |
| `slug` | 这是网址。翻了网址就断，中文页会 404 |
| `tags` | 这些生成标签页路由。翻了会造出重复的垃圾页面 |
| `authors` | 这是账号 ID，不是名字 |
| `image` | 这是文件路径 |
| `date` | 日期 |

### 必须翻成中文的字段

| 字段 | 说明 |
| --- | --- |
| `title` | 文章标题 |
| `description` | 摘要，会显示在搜索结果里 |
| `category` | 用下面的固定对照表，不要自己发挥 |
| `article_type` | 用下面的固定对照表，不要自己发挥 |

### `category` 固定译法（照抄，不要改）

```
Plant phenotyping      → 植物表型
AI & machine learning  → 人工智能与机器学习
Imaging & 3D           → 成像与三维
Developer tools        → 开发者工具
Research practice      → 科研实践
```

### `article_type` 固定译法（照抄，不要改）

```
Technical guide  → 技术指南
Research project → 研究项目
Workflow         → 工作流
Reference        → 参考资料
```

### 正确示范

英文原文：

```yaml
---
slug: root-quantify
title: "Root Quantify: Interactive Root Image Preprocessing in Python"
description: "A workflow for preprocessing root images."
authors: [liangchao]
category: Plant phenotyping
article_type: Technical guide
tags: [plant-phenotyping, image-analysis, python]
image: /img/blog-default.jpg
---
```

正确的中文版：

```yaml
---
slug: root-quantify
title: "Root Quantify：用 Python 交互式预处理根系图像"
description: "一套根系图像预处理的工作流。"
authors: [liangchao]
category: 植物表型
article_type: 技术指南
tags: [plant-phenotyping, image-analysis, python]
image: /img/blog-default.jpg
---
```

注意看：`slug`、`tags`、`authors`、`image` 四个**完全没动**。只有 `title`、`description`、`category`、`article_type` 变成了中文。

### 错误示范（绝对不要这样）

```yaml
slug: 根系量化              ← 错！slug 永远不能翻
tags: [植物表型, 图像分析]   ← 错！tags 永远不能翻
authors: [邓良超]           ← 错！authors 是账号 ID
```

---

## 规则二：正文里这些东西不要翻

1. **代码块内容不要翻。** 三个反引号包起来的内容原样保留，命令、代码、参数名一个字符都不要动。

   ```bash
   git clone https://github.com/OpenDroneMap/WebODM
   ```

   ↑ 这整块原样复制。**唯一例外**：代码块里的英文*注释*可以翻。

2. **网址原样保留。** `https://...` 不要动。但链接的**显示文字**要翻：

   - 英文：`[QGIS documentation](https://docs.qgis.org/)`
   - 中文：`[QGIS 文档](https://docs.qgis.org/)` ← 只翻方括号里的字

3. **`<!-- truncate -->` 这一行原样保留，位置也不要挪。** 它控制列表页显示多少摘要。丢了会导致整篇文章都灌进列表页。

4. **`:::caution`、`:::note`、`:::tip` 这类标记原样保留**，但它后面的标题文字和里面的内容要翻：

   ```
   :::caution 本工作流对版本敏感

   模型名称和 API 变化很快……

   :::
   ```

   ↑ `:::caution` 和结尾的 `:::` 不动，其余翻译。

5. **表格的 `| --- | --- |` 分隔行不要动**，表头和单元格内容要翻。

6. **专业术语保留英文**：NDVI、LoRA、GPU、CUDA、PyTorch、NoData、EXIF、CRS、GLCM 等。已经通用的英文缩写不要硬翻成中文。

---

## 规则三：翻译风格

**忠实翻译。** 贴近英文原意，只做必要的中文表达调整。

- 不要增加原文没有的内容
- 不要删减原文有的内容
- 不要改变段落结构、标题层级、列表项数量
- 原文有 10 条要点，译文也必须是 10 条

---

## 做完后必须自查

在项目根目录运行：

```bash
node scripts/check-blog-i18n.mjs
```

这个脚本会逐篇对比中英文，把错误直接列出来。**必须做到输出里没有任何 `✗` 才算完成。**

它会抓这些错误：

- slug 被改了
- tags 被翻译了
- `<!-- truncate -->` 丢了
- 代码块反引号数量对不上（少写了结尾的三个反引号）
- 正文里根本没有中文（说明忘了翻，还是英文）

如果报错，按提示修，再跑一遍，直到干净。然后跑构建确认没坏：

```bash
npm run build
```

---

## 参考样例

这两个文件是已经做好的标准样例，格式照着它们来：

- `i18n/zh-Hans/docusaurus-plugin-content-blog/2023-04-22-dji-p4m-webodm-qgis-tutorial.md`
- `i18n/zh-Hans/docusaurus-plugin-content-blog/2024-01-11-local-llm-training-guide.md`

开始前先读一遍其中一个，对照它的英文原文 `blog/2023-04-22-dji-p4m-webodm-qgis-tutorial.md`，看清楚哪些翻了、哪些没翻。

---

## 最后提醒

不确定某个东西要不要翻的时候，**不要自己猜**。规则是：

> **能被机器读的 → 不翻**（slug、tags、authors、image、date、代码、网址）
>
> **给人看的 → 翻**（title、description、category、article_type、正文）
