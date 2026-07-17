# PPTX 翻译技能（中英双向翻译）

一个 AI 辅助的演示文稿工具，用于在英文和中文之间翻译 PowerPoint 幻灯片，同时保留布局、样式和图片。

本教程展示如何配置、使用和集成 **PPTX 翻译技能**，以在不丢失布局完整性或排版的情况下无缝翻译演示文稿。

它支持双向翻译（`zh2en` 和 `en2zh`），并直接与 AI 智能体平台集成。

---

## 1. 概述

翻译幻灯片可能很繁琐，因为文本扩展经常会破坏布局容器、字号和换行。

此工具通过在段落级别翻译文本块（保留文本运行）、自动清理目标标点、智能替换字体以及运行自动格式一致性质量检查来解决此问题。

---

## 2. GitHub 仓库链接

直接从仓库访问翻译脚本和模板：
[pptx-zh2en](https://github.com/smiler488/pptx-zh2en)

---

## 3. 如何安装和配置

按照以下步骤设置翻译工具：
1. 安装 Python 依赖：
   ```bash
   pip install python-pptx
   ```
2. 将仓库克隆到 IDE 的活跃技能文件夹：
   * **Cursor**: `.cursor/skills/pptx-zh2en`
   * **CodeBuddy**: `/plugin marketplace add smiler488/pptx-zh2en`

---

## 4. 如何使用和运行翻译

你可以使用命令行独立运行翻译脚本：
1. **提取源文本：**
   ```bash
   python scripts/translate_pptx_inline.py --mode extract --direction zh2en -i source.pptx -t trans.json
   ```
2. **翻译条目：** 编辑 `trans.json` 并填写翻译字段。
3. **写回翻译文本：**
   ```bash
   python scripts/translate_pptx_inline.py --mode apply --direction zh2en -i source.pptx -t trans.json -o target.pptx
   ```

---

- 仓库：https://github.com/smiler488/pptx-zh2en
- 作者：Liangchao Deng (Shihezi University / CAS-CEMPS)
- 工具名称：PPTX translation helper
- 许可证：CC BY-SA 3.0
- 用途：为学术和研究演示幻灯片提供自动化的高保真翻译。
