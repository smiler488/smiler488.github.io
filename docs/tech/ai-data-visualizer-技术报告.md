# AI Data Visualizer 中文技术报告

## 文档目的
- 面向研发与使用者，阐述 AI Data Visualizer 的架构设计、核心实现与使用方法。
- 便于对外汇报“如何实现”“如何正确使用”，以及常见问题的定位与扩展方向。

## 架构总览
- 前端单页应用，基于 Docusaurus 页面容器与 React。
- 图表渲染使用 Apache ECharts（通过 CDN 动态加载）。
- 无服务端存储；浏览器内完成 CSV/TSV 解析、提示词构造、调用第三方大模型 API、并渲染图表。
- 关键技术点：
  - 数据摘要生成与提示词（prompt）模板化
  - 统一 JSON Schema 的 AI 输出解析与修复
  - ECharts option 归一化、误差棒与 Tukey 字母支持

## 关键模块与代码位置
- ECharts加载：`src/pages/app/ai-data-visualizer/index.js:46-75`
- AI端点选择：`getTargetApiUrl` `src/pages/app/ai-data-visualizer/index.js:974-982`
- 分析流程主入口：`handleAnalyze` `src/pages/app/ai-data-visualizer/index.js:984-1048`
- 提示词构造：`buildVisualizationPrompt` `src/pages/app/ai-data-visualizer/index.js:268-299`
- 响应提取：`extractAssistantText` `src/pages/app/ai-data-visualizer/index.js:258-266`
- JSON解析与修复：`tryParseJson`、`cleanupJsonText`、`attemptJsonRepair`
  - `src/pages/app/ai-data-visualizer/index.js:81-122, 124-143`
- 结果处理：`processAiText` `src/pages/app/ai-data-visualizer/index.js:1050-1085`
- 图表下载：`handleDownloadChart` `src/pages/app/ai-data-visualizer/index.js:1087-1098`

## 数据流与错误处理
- 文件读取：用户上传 CSV/TSV，浏览器解析，生成 `datasetSummary` 与分类/数值提示。
- 提示词生成：将数据摘要与用户目标（Analysis goal）拼装为严格 JSON Schema 指令的 prompt。
- API调用：
  - 默认端点（可内置演示端点）或自定义端点（需填写 URL、Bearer Token）。
  - 响应体解析规则：优先取 `choices[0].message.content`（或 HunYuan 兼容字段），否则回退为原文或字符串化对象。
- JSON解析：
  - 先做文本清理与简单修复（代码块围栏、中文引号、尾逗号、单引号键等）。
  - 若失败，尝试截取首尾花括号片段并再次修复；仍失败则报错。
- 错误提示：
  - 非法JSON：抛出 “AI response was not valid JSON. Ask it to follow the schema.”（`src/pages/app/ai-data-visualizer/index.js:1054`）。
  - 缺少 `chart_option` 或结构非法：设置 `chartError` 并提示重试。
  - 网络或鉴权失败：显示状态并可回退到 mock 响应（默认端点不可用时）。

## AI输出Schema（必须遵守）
- 固定字段：
  - `summary`：字符串，2-3 句的自然语言总结
  - `insights`：字符串数组，若干要点
  - `chart_option`：合法的 ECharts option 对象
- 额外约定：
  - 若涉及 ANOVA/Tukey：`tukey_letters` 为 JSON 对象，键为类目名称，值为字母（如 `{"Treatment_A":"a"}`）。
  - 若需不确定性：返回 `error_bars` 数组 `[{"name":"A","low":10,"high":12}]`，由前端渲染为自定义胡须；不要在 JSON 中包含函数或注释。
- 模板来源：`buildVisualizationPrompt` `src/pages/app/ai-data-visualizer/index.js:283-299`

## 图表归一化与增强
- 归一化流程：`processAiText` 中将 AI 的 `chart_option` 转换为规范 ECharts 配置，并附加必要项（title/tooltip/grid/axes/visualMap/series）。
- 误差棒：若 `error_bars` 有值，调用附加函数生成 whiskers 形态（代码参考归一化工具函数）。
- Tukey字母：若提供 `tukey_letters`，将结果叠加为标注层或类目标签。

## 使用指南（面向普通用户）
- 打开页面：`/app/ai-data-visualizer`
- 上传数据：CSV/TSV（表头在首行，内容 tidy rows）。
- 设置分析目标（Analysis goal）：用简洁英文或中文描述你要看什么，例如“比较各处理组在不同季节的差异并给出Tukey字母”。
- 选择端点：
  - 勾选 “Use built-in HunYuan endpoint” 走内置演示端点；
  - 或填入自定义 `API URL` 与 `Bearer token`（将仅在浏览器内使用，不会存储）。
- 生成图表：点击 “Generate visualization”。若成功，右侧显示预览、要点与原始 AI 文本。
- 下载图片：点击 “Download PNG”。

## 常见问题与解决方案
- 一直提示 “AI response was not valid JSON”：
  - 将分析目标中加入硬约束：“只返回严格 JSON，字段为 summary、insights、chart_option；不要额外文字或Markdown”。
  - 确认你的自定义端点把 JSON 字符串放在 `choices[0].message.content`；不要返回对象或混杂解释。
- 图表空白或报错：
  - 检查 `chart_option` 内 `series` 与 `xAxis/yAxis` 类目长度是否一致。
  - 简化图表类型（先用柱状或折线），逐步增加可视化复杂度。
- 端点不可用：
  - 默认端点 403/404/405 时将回退到 `mock://ai-data-visualizer`（`src/pages/app/ai-data-visualizer/index.js:1027-1033`）。

## 安全与隐私
- 浏览器内运行，不上传数据到站点服务器；自定义密钥只保留在本地内存。
- 内置端点与演示密钥仅用于演示与联调，不适合生产；建议使用自己的代理端点并在浏览器中填写 Bearer Token（不要在代码中硬编码）。

## 二次开发建议
- 增强容错：
  - 提升 JSON 修复策略（多段文本抽取、字段名兼容、Markdown剥离）。
  - 对 `chart_option` 自动补全缺失字段与默认样式。
- 可视化扩展：
  - 支持更多图表类型（boxplot、sankey、treemap 等已内置），优化默认配色与图例布局。
- 国际化：
  - 将界面与提示信息抽取为 i18n 资源。

## 版本与维护
- 页面入口：`src/pages/app/ai-data-visualizer/index.js`
- 依赖：ECharts CDN（`echarts@5.x`）。
- 构建与预览：
  - 构建：`npm run build`
  - 本地预览：`npm run serve` 后访问 `http://localhost:3000/`
