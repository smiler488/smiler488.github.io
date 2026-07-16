# PPTX Translation Skill (中英双向翻译)

An AI-assisted presentation utility to translate PowerPoint decks between English and Chinese while retaining layouts, styles, and images.

This tutorial shows how to configure, use, and integrate the **PPTX Translation Skill** to translate presentations seamlessly without losing layout integrity or typography.

It supports dual-direction translation (`zh2en` and `en2zh`) and integrates directly with AI agent platforms.

---

## 1. Overview

Translating slides can be tedious because text expansion often breaks layout containers, font sizes, and line-breaks.

This utility addresses this issue by translating text blocks at the paragraph level (preserving text runs), cleaning target punctuation automatically, replacing fonts intelligently, and running automated quality checks for formatting consistency.

---

## 2. GitHub Repository Link

Access the translation script and templates directly from the repository:
[pptx-zh2en](https://github.com/smiler488/pptx-zh2en)

---

## 3. How to Install and Configure

Follow these steps to set up the translation tool:
1. Install Python dependencies:
   ```bash
   pip install python-pptx
   ```
2. Clone the repository to your IDE's active skills folder:
   * **Cursor**: `.cursor/skills/pptx-zh2en`
   * **CodeBuddy**: `/plugin marketplace add smiler488/pptx-zh2en`

---

## 4. How to Use & Run Translation

You can run the translation script independently using the command line:
1. **Extract source text:**
   ```bash
   python scripts/translate_pptx_inline.py --mode extract --direction zh2en -i source.pptx -t trans.json
   ```
2. **Translate entries:** Edit `trans.json` and fill in the translated fields.
3. **Write back translated text:**
   ```bash
   python scripts/translate_pptx_inline.py --mode apply --direction zh2en -i source.pptx -t trans.json -o target.pptx
   ```

---

- Repository: https://github.com/smiler488/pptx-zh2en
- Author: Liangchao Deng (Shihezi University / CAS-CEMPS)
- Tool Name: PPTX translation helper
- License: CC BY-SA 3.0
- Purpose: Automated high-fidelity translation for academic and research presentation slides.
