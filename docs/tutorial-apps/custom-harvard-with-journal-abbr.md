# Custom Harvard With Journal Abbr (CSL)

A Zotero/Mendeley CSL style that formats in-text citations as (Author, Journal Abbr, Year) and uses journal abbreviations.
# Custom Zotero Citation Style for Academic Presentations

This tutorial shows how to create, install, and configure a **custom Zotero citation style** for generating concise, standardized references in PowerPoint or Beamer presentations.  
It is based on a *Harvard-style format with journal abbreviations*, producing outputs such as:

> (Deng et al., Plant Phenomics, 2025)

This format keeps your slides clean and professional while ensuring consistency with academic citation standards.

---

## 1. Overview

When preparing academic presentations, full-length reference lists can clutter your slides.  
A compact citation format like `(Author, Journal Abbr, Year)` conveys all essential information clearly and efficiently.

This tutorial introduces a custom **CSL (Citation Style Language)** file you can import into Zotero to automatically generate citations in this format.


---

## 2. CSL Template Code

Save the following XML code as **`custom-harvard-with-journal-abbr.csl`** on your computer:
[custom-harvard-with-journal-abbr.csl](https://github.com/smiler488/custom-harvard-with-journal-abbr)

---

## 3.How to Install and Configure

Follow these steps to install and enable the custom style in Zotero:
	1.	Open Zotero
	2.	Go to Preferences → Cite → Style Manager
	3.	Click “+” → Add Style…
	4.	Select the file custom-harvard-with-journal-abbr.csl
	5.	The new style will appear in the list as
Custom Harvard With Journal Abbr
	6.	To configure for exporting:
      Go to Preferences → Export → Item Format, Select 'Custom Harvard With Journal Abbr'
      

---

## 4. How to Use in PowerPoint or Beamer

After setup, Zotero allows you to insert references directly into slides:
	1.	In Zotero, select one or more items.
	2.	Drag the selected items directly into a PowerPoint.
	3.	Zotero will automatically insert formatted citations, e.g.:
```
(Deng et al., Plant Phenomics, 2025)
```
For multiple references:
```
(Tang et al., J. Integr. Agric., 2025).
(Deng et al., Plant Phenomics, 2025).
```

- Style ID: http://www.zotero.org/styles/custom-harvard-with-journal-abbr
- Author: Liangchao Deng (Shihezi University / CAS-CEMPS)
- Style Name: Custom Harvard With Journal Abbr
- License: CC BY-SA 3.0￼
- Purpose: To streamline citation insertion and improve presentation aesthetics in academic contexts.
