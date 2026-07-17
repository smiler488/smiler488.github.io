# 带期刊缩写的自定义 Harvard 引用格式（CSL）

一种 Zotero/Mendeley CSL 样式，将文中引用格式化为 (作者, 期刊缩写, 年份)，并使用期刊缩写。
# 用于学术报告的自定义 Zotero 引用样式

本教程展示如何创建、安装和配置一个**自定义 Zotero 引用样式**，以便在 PowerPoint 或 Beamer 演示文稿中生成简洁、标准化的参考文献。
它基于*带期刊缩写的 Harvard 格式*，生成如下输出：

> (Deng et al., Plant Phenomics, 2025)

此格式使幻灯片保持简洁专业，同时确保与学术引用标准的一致性。

---

## 1. 概述

在准备学术报告时，完整的参考文献列表会使幻灯片显得杂乱。
像 `(作者, 期刊缩写, 年份)` 这样的紧凑引用格式可以清晰高效地传达所有关键信息。

本教程介绍一个自定义 **CSL（Citation Style Language）** 文件，你可以将其导入 Zotero 以自动生成此格式的引用。


---

## 2. CSL 模板代码

将以下 XML 代码保存为计算机上的 **`custom-harvard-with-journal-abbr.csl`** 文件：
[custom-harvard-with-journal-abbr.csl](https://github.com/smiler488/custom-harvard-with-journal-abbr)

---

## 3. 如何安装和配置

按照以下步骤在 Zotero 中安装和启用自定义样式：
	1.	打开 Zotero
	2.	进入 Preferences → Cite → Style Manager
	3.	点击 "+" → Add Style…
	4.	选择文件 custom-harvard-with-journal-abbr.csl
	5.	新样式将以以下名称出现在列表中
Custom Harvard With Journal Abbr
	6.	配置导出：
      进入 Preferences → Export → Item Format，选择 'Custom Harvard With Journal Abbr'
      

---

## 4. 如何在 PowerPoint 或 Beamer 中使用

设置完成后，Zotero 允许你将参考文献直接插入幻灯片：
	1.	在 Zotero 中，选择一个或多个条目。
	2.	将所选条目直接拖入 PowerPoint。
	3.	Zotero 会自动插入格式化后的引用，例如：
```
(Deng et al., Plant Phenomics, 2025)
```
多条引用：
```
(Tang et al., J. Integr. Agric., 2025).
(Deng et al., Plant Phenomics, 2025).
```

- 样式 ID：http://www.zotero.org/styles/custom-harvard-with-journal-abbr
- 作者：Liangchao Deng (Shihezi University / CAS-CEMPS)
- 样式名称：Custom Harvard With Journal Abbr
- 许可证：CC BY-SA 3.0￼
- 用途：简化引用插入并改善学术场景中的演示美观度。
