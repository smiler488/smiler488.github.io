# Skill Creator & Publisher

A developer utility to automatically create standard AI agent skills and publish them directly to GitHub repositories.

This tutorial shows how to configure, use, and verify the **Skill Creator & Publisher** tool to streamline the development and publishing of custom capabilities for AI agents.

It supports mainstream coding agents including CodeBuddy, Claude Code, Cursor, Cline, and Windsurf.

---

## 1. Overview

When developing custom features (Skills) for agentic workflows, setting up folders, configuration lists, helper scripts, and publishing tasks can be repetitive.

This tool automates the process by generating standard skeleton configurations (`SKILL.md`, `skill.json`, template scripts, marketplace lists) and using the GitHub CLI to publish them as modular, reusable assets.

---

## 2. GitHub Repository Link

Access the code, scripts, and template configurations directly from the repository:
[skill-creator-publish](https://github.com/smiler488/skill-creator-publish)

---

## 3. How to Install and Configure

Follow these steps to configure and use the tool:
1. Log in to the GitHub CLI:
   ```bash
   gh auth login
   ```
2. Clone the repository into your agent's active skills directory depending on your IDE setup:
   * **Cursor**: `.cursor/skills/skill-creator-publish`
   * **Claude Code**: `.claude/skills/skill-creator-publish`
   * **Cline**: `.cline/skills/skill-creator-publish`
   * **CodeBuddy**: `/plugin marketplace add smiler488/skill-creator-publish`

---

## 4. How to Use & Generate Skills

You can run the script independently to generate a new skill structure:
1. Navigate to the tool's folder and run the python script:
   ```bash
   python scripts/create_skill.py --name my-skill --description "Description" --author "Author" --keywords "k1,k2"
   ```
2. The script creates the complete skeleton ready for development and deployment.

---

- Repository: https://github.com/smiler488/skill-creator-publish
- Author: Liangchao Deng (Shihezi University / CAS-CEMPS)
- Tool Name: Skill Creator & Publisher
- License: CC BY-SA 3.0
- Purpose: Streamlining structure creation and automated publishing for agentic IDE skills.
