# 技能创建与发布器

一个开发者工具，用于自动创建标准 AI 智能体技能并直接发布到 GitHub 仓库。

本教程展示如何配置、使用和验证 **技能创建与发布器** 工具，以简化 AI 智能体自定义能力的开发和发布。

它支持主流编码智能体，包括 CodeBuddy、Claude Code、Cursor、Cline 和 Windsurf。

---

## 1. 概述

在为智能体工作流开发自定义功能（技能）时，设置文件夹、配置清单、辅助脚本和发布任务可能是重复性的。

此工具通过生成标准骨架配置（`SKILL.md`、`skill.json`、模板脚本、市场清单）并使用 GitHub CLI 将其发布为模块化、可复用的资产来自动化此过程。

---

## 2. GitHub 仓库链接

直接从仓库访问代码、脚本和模板配置：
[skill-creator-publish](https://github.com/smiler488/skill-creator-publish)

---

## 3. 如何安装和配置

按照以下步骤配置和使用该工具：
1. 登录 GitHub CLI：
   ```bash
   gh auth login
   ```
2. 根据你的 IDE 设置，将仓库克隆到智能体的活跃技能目录：
   * **Cursor**: `.cursor/skills/skill-creator-publish`
   * **Claude Code**: `.claude/skills/skill-creator-publish`
   * **Cline**: `.cline/skills/skill-creator-publish`
   * **CodeBuddy**: `/plugin marketplace add smiler488/skill-creator-publish`

---

## 4. 如何使用和生成技能

你可以独立运行脚本以生成新的技能结构：
1. 导航到工具文件夹并运行 Python 脚本：
   ```bash
   python scripts/create_skill.py --name my-skill --description "Description" --author "Author" --keywords "k1,k2"
   ```
2. 脚本会创建完整的骨架，可供开发和部署。

---

- 仓库：https://github.com/smiler488/skill-creator-publish
- 作者：Liangchao Deng (Shihezi University / CAS-CEMPS)
- 工具名称：Skill Creator & Publisher
- 许可证：CC BY-SA 3.0
- 用途：简化智能体 IDE 技能的结构创建和自动发布。
