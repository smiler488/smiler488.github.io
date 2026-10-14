/** Chinese interface text for this tool (src/lib/i18n/toolText.js). */
export default {
  Measurement: "测量值",
  "Factor / code": "因子 / 编码",
  "Index / sequence": "序号 / 序列",
  Text: "文本",
  "Needs review": "需复核",
  Note: "提示",
  "Why this matters and what to rule out": "为何值得注意、需先排除什么",
  "Why it is unusual:": "为何不寻常：",
  "Ordinary explanations:": "常见解释：",
  "Hide cells": "收起单元格",
  "Show cells": "查看单元格",
  Row: "行",
  Line: "行",
  "First 30 of {0} rows; all are in the report.":
    "仅显示 {0} 行中的前 30 行；报告中包含全部。",
  "Choose a data file. It is read in this browser and never uploaded.":
    "选择一个数据文件。文件只在本浏览器中读取，不会上传。",
  "{0} rows and {1} columns read; {2} measurement columns selected. Check the selection, then run the screen.":
    "已读取 {0} 行、{1} 列，选中了 {2} 个测量值列。请检查选择，然后开始筛查。",
  "The file is larger than 50 MB.": "文件大于 50 MB。",
  "Reading file…": "正在读取文件…",
  "The file could not be read: {0}": "无法读取文件：{0}",
  "{0} pattern(s) need review and {1} are noted. Each lists its cells and the ordinary explanations to rule out.":
    "{0} 处模式需复核，另有 {1} 处提示。每一处都列出了相关单元格和需先排除的常见解释。",
  "No pattern needs review with these settings ({0} noted).":
    "在当前设置下没有需复核的模式（提示 {0} 处）。",
  "Screen a research data file for patterns that are unusual in genuine measurements — duplicated or near-identical records, pasted runs of values, uneven final digits, equal increments, impossible means — and export a report that a reviewer can check cell by cell.":
    "对科研数据文件筛查真实测量中少见的模式——重复或近似重复的记录、粘贴的数值序列、末位数字分布不均、等步长、不可能的均值——并导出可逐个单元格复核的报告。",
  "A flagged pattern is a reason to look at the raw records, not proof of misconduct. Each finding lists the ordinary explanations to rule out first.":
    "被标出的模式只是需要查看原始记录的理由，并不证明存在不端行为。每项发现都列出了需先排除的常见解释。",
  "01 · Data file": "01 · 数据文件",
  "Choose a file": "选择文件",
  "Or try an example: a synthetic field trial with planted problems":
    "或试用示例：一份植入了问题的模拟田间试验数据",
  "CSV, TSV, TXT (delimited or space-separated), Excel or JSON, up to 50 MB. Numbers in text files are read exactly as written; spreadsheet numbers as stored, not as displayed.":
    "支持 CSV、TSV、TXT（分隔符或空格分隔）、Excel 或 JSON，最大 50 MB。文本文件中的数字按原样读取；电子表格中的数字按存储值读取，而非显示格式。",
  Sheet: "工作表",
  "02 · Columns": "02 · 列",
  "What to screen": "筛查哪些列",
  "Measurement columns are screened. Index columns (1, 2, 3 …) and factor codes (treatment, replicate) are left out because their regular patterns are by design; tick or untick to change.":
    "只筛查测量值列。序号列（1、2、3……）和因子编码（处理、重复）的规律是设计使然，因此不参与筛查；可勾选或取消勾选调整。",
  Screen: "筛查",
  Column: "列",
  "Detected as": "识别为",
  Numbers: "数值个数",
  Distinct: "不同值个数",
  "Table of reported means? Check them with GRIM": "是汇总均值表？用 GRIM 检查",
  "Mean column": "均值列",
  "Sample size (n) column": "样本量（n）列",
  "Items per participant": "每人题目数",
  "GRIM applies to means of integer data (counts, Likert items) and to n below 10 to the power of the reported decimals.":
    "GRIM 适用于整数数据（计数、李克特量表题）的均值，且要求 n 小于 10 的“报告小数位数”次方。",
  "03 · Settings": "03 · 设置",
  Thresholds: "阈值",
  "Review level (q and expected chance occurrences)":
    "复核水平（q 值与偶然出现期望次数）",
  "Shortest repeated run": "最短重复序列长度",
  "Shortest run of equal steps": "最短等步长序列长度",
  "Outlier |modified z|": "离群值 |修正 z|",
  "Defaults were validated by simulation: on clean data of six kinds, 0–2% of datasets got a finding to review.":
    "默认值经过模拟验证：在六类正常数据中，有 0–2% 的数据集出现需复核的发现。",
  "Run the screen": "开始筛查",
  "04 · Findings": "04 · 发现",
  Results: "结果",
  "{0} columns screened, {1} digit tests run":
    "筛查了 {0} 列，进行了 {1} 项末位数字检验",
  "Download report (HTML)": "下载报告（HTML）",
  "Findings (JSON)": "发现（JSON）",
  "Flagged cells (CSV)": "被标出的单元格（CSV）",
  "The report records the file's SHA-256, the columns and thresholds used and the tool version, so anyone can reproduce it; open it in a browser and print to PDF if needed.":
    "报告记录了文件的 SHA-256、所用的列和阈值以及工具版本，任何人都可以复现；可在浏览器中打开并打印为 PDF。",
  "No patterns found.": "未发现任何模式。",
  "05 · Optional": "05 · 可选",
  "Plain-language summary by AI": "AI 生成通俗摘要",
  "Only the finding descriptions above are sent to the model you choose, never the data file. The summary is added to the report, marked as AI-written.":
    "只会把上方的发现描述发送给你选择的模型，绝不发送数据文件。摘要会加入报告，并标明由 AI 撰写。",
  "Summary model": "摘要模型",
  "Writing…": "撰写中…",
  "Write summary": "撰写摘要",
};
