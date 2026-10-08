import React, { useState } from "react";
import Heading from "@theme/Heading";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./styles.module.css";

const DOI = "10.5281/zenodo.17544584";
const DOI_URL = `https://doi.org/${DOI}`;

const APA =
  `LiangchaoDeng. (2025). smiler488/smiler488.github.io: Digital Plant Phenotyping Platform v25.0 (v25.0.0). Zenodo. ${DOI_URL}`;

const BIBTEX = `@software{Deng2025_DPPP_v25,
  author       = {Deng, Liangchao},
  title        = {Digital Plant Phenotyping Platform (v25.0)},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {${DOI}},
  url          = {${DOI_URL}},
  note         = {[Computer software]}
}`;

const COPY = {
  en: {
    title: "Cite this work",
    intro:
      "If you use the Digital Plant Phenotyping Platform v25.0 or any of its tools in your research, please cite it.",
    apa: "APA",
    bibtex: "BibTeX",
    zotero: "In Zotero, open the File menu and choose Import from Clipboard.",
    copyApa: "Copy citation",
    copyBib: "Copy BibTeX",
    copied: "Copied",
  },
  zh: {
    title: "引用本工作",
    intro:
      "如果你在研究中使用了数字植物表型平台 v25.0 或其中的任一工具，请引用。",
    apa: "APA",
    bibtex: "BibTeX",
    zotero: "在 Zotero 中打开“文件”菜单，选择“从剪贴板导入”。",
    copyApa: "复制引用",
    copyBib: "复制 BibTeX",
    copied: "已复制",
  },
};

async function copyText(text) {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "absolute";
  area.style.left = "-9999px";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  document.body.removeChild(area);
}

function CopyButton({ text, label, copiedLabel }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await copyText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };
  return (
    <button
      type="button"
      className={styles.copyButton}
      onClick={onCopy}
      aria-live="polite"
    >
      {copied ? copiedLabel : label}
    </button>
  );
}

export default function CitationNotice() {
  const { i18n } = useDocusaurusContext();
  const t = i18n.currentLocale === "zh-Hans" ? COPY.zh : COPY.en;

  return (
    <section className={styles.citation} aria-labelledby="cite-this-work">
      <header className={styles.header}>
        <div>
          <Heading as="h2" id="cite-this-work" className={styles.title}>
            {t.title}
          </Heading>
          <p className={styles.intro}>{t.intro}</p>
        </div>
        <Link className={styles.doi} to={DOI_URL}>
          <span className={styles.doiLabel}>DOI</span>
          <span>{DOI}</span>
        </Link>
      </header>

      <div className={styles.columns}>
        <div className={styles.column}>
          <div className={styles.columnHead}>
            <span className={styles.label}>{t.apa}</span>
            <CopyButton text={APA} label={t.copyApa} copiedLabel={t.copied} />
          </div>
          <p className={styles.apa}>
            LiangchaoDeng. (2025).{" "}
            <em>
              smiler488/smiler488.github.io: Digital Plant Phenotyping Platform
              v25.0 (v25.0.0)
            </em>
            . Zenodo. <Link to={DOI_URL}>{DOI_URL}</Link>
          </p>
        </div>

        <div className={styles.column}>
          <div className={styles.columnHead}>
            <span className={styles.label}>{t.bibtex}</span>
            <CopyButton text={BIBTEX} label={t.copyBib} copiedLabel={t.copied} />
          </div>
          <pre className={styles.code}>
            <code>{BIBTEX}</code>
          </pre>
          <p className={styles.hint}>{t.zotero}</p>
        </div>
      </div>
    </section>
  );
}
