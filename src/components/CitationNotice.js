import React, { useState } from "react";

const containerBaseStyle = {
  maxWidth: "800px",
  margin: "60px auto 0",
  textAlign: "left",
  padding: "0 20px",
};

const headingStyle = {
  fontSize: "2rem",
  fontWeight: "600",
  marginBottom: "16px",
  color: "var(--ifm-color-emphasis-900)",
};

const paragraphStyle = {
  fontSize: "1.1rem",
  color: "var(--ifm-color-emphasis-700)",
  lineHeight: 1.6,
  marginBottom: "16px",
};

const blockquoteStyle = {
  borderLeft: "4px solid var(--ifm-border-color)",
  paddingLeft: "16px",
  color: "var(--ifm-color-emphasis-700)",
  fontStyle: "italic",
  lineHeight: 1.6,
  marginBottom: "16px",
};

const codeBlockStyle = {
  backgroundColor: "var(--ifm-background-surface-color)",
  padding: "16px",
  borderRadius: "12px",
  overflowX: "auto",
  border: "1px solid var(--ifm-border-color)",
  color: "var(--ifm-color-emphasis-800)",
};

const subheadingStyle = {
  fontSize: "1.5rem",
  fontWeight: "600",
  marginBottom: "16px",
  color: "var(--ifm-color-emphasis-900)",
};

export default function CitationNotice({ containerStyle }) {
  const [copied, setCopied] = useState(false);
  const [bibCopied, setBibCopied] = useState(false);

  const apaCitation =
    "LiangchaoDeng. (2025). smiler488/smiler488.github.io: Digital Plant Phenotyping Platform v25.0 (v25.0.0). Zenodo. https://doi.org/10.5281/zenodo.17544584";

  const handleCopyCitation = async () => {
    try {
      if (
        typeof navigator !== "undefined" &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(apaCitation);
      } else if (typeof document !== "undefined") {
        // Fallback copy method for older browsers
        const textarea = document.createElement("textarea");
        textarea.value = apaCitation;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "absolute";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  const bibtexCitation = `@software{Deng2025_DPPP_v25,
  author       = {Deng, Liangchao},
  title        = {Digital Plant Phenotyping Platform (v25.0)},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17544584},
  url          = {https://doi.org/10.5281/zenodo.17544584},
  note         = {[Computer software]}
}`;

  const handleCopyBibtex = async () => {
    try {
      if (
        typeof navigator !== "undefined" &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(bibtexCitation);
      } else if (typeof document !== "undefined") {
        const textarea = document.createElement("textarea");
        textarea.value = bibtexCitation;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "absolute";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      setBibCopied(true);
      setTimeout(() => setBibCopied(false), 2000);
    } catch {
      setBibCopied(false);
    }
  };

  return (
    <div style={{ ...containerBaseStyle, ...containerStyle }}>
      <h2 style={headingStyle}>Citation</h2>
      <p style={paragraphStyle}>
        If you use <strong>Digital Plant Phenotyping Platform v25.0</strong> or
        any of its applications in your research, please cite it as:
      </p>
      <blockquote style={blockquoteStyle}>
        LiangchaoDeng. (2025).{" "}
        <em>
          smiler488/smiler488.github.io: Digital Plant Phenotyping Platform
          v25.0 (v25.0.0)
        </em>
        . Zenodo.<br />
        <a href="https://doi.org/10.5281/zenodo.17544584">
          https://doi.org/10.5281/zenodo.17544584
        </a>
      </blockquote>
      <div style={{ display: "flex", alignItems: "center", marginBottom: "16px", gap: "12px" }}>
        <button
          type="button"
          onClick={handleCopyCitation}
          style={{
            padding: "8px 16px",
            borderRadius: "8px",
            border: "1px solid var(--ifm-color-emphasis-900)",
            backgroundColor: "var(--ifm-color-emphasis-900)",
            color: "var(--ifm-color-emphasis-0)",
            cursor: "pointer",
            fontSize: "0.95rem",
            fontWeight: 500,
            transition: "transform 0.15s ease, background-color 0.15s ease",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.backgroundColor = "var(--ifm-color-primary)";
            e.currentTarget.style.borderColor = "var(--ifm-color-primary)";
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.backgroundColor = "var(--ifm-color-emphasis-900)";
            e.currentTarget.style.borderColor = "var(--ifm-color-emphasis-900)";
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          {copied ? "Copied!" : "Copy citation"}
        </button>
        <span
          style={{
            fontSize: "0.9rem",
            color: copied
              ? "var(--ifm-color-success)"
              : "var(--ifm-color-emphasis-600)",
          }}
        >
          {copied ? "Citation copied to clipboard." : "Click to copy APA citation."}
        </span>
      </div>
      <a
        href="https://doi.org/10.5281/zenodo.17544584"
        style={{ display: "inline-block", marginBottom: "24px" }}
      >
        <img
          src="https://zenodo.org/badge/DOI/10.5281/zenodo.17544584.svg"
          alt="DOI badge"
          style={{ height: "32px" }}
        />
      </a>
      <hr
        style={{
          border: "none",
          borderTop: "1px solid var(--ifm-border-color)",
          marginBottom: "24px",
        }}
      />
      <h3 style={subheadingStyle}>
        BibTeX citation (Zotero → File → Import from Clipboard)
      </h3>
      <div style={{ display: "flex", alignItems: "center", marginBottom: "16px", gap: "12px" }}>
        <button
          type="button"
          onClick={handleCopyBibtex}
          style={{
            padding: "8px 16px",
            borderRadius: "8px",
            border: "1px solid var(--ifm-color-emphasis-900)",
            backgroundColor: "var(--ifm-color-emphasis-900)",
            color: "var(--ifm-color-emphasis-0)",
            cursor: "pointer",
            fontSize: "0.95rem",
            fontWeight: 500,
            transition: "transform 0.15s ease, background-color 0.15s ease",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.backgroundColor = "var(--ifm-color-primary)";
            e.currentTarget.style.borderColor = "var(--ifm-color-primary)";
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.backgroundColor = "var(--ifm-color-emphasis-900)";
            e.currentTarget.style.borderColor = "var(--ifm-color-emphasis-900)";
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          {bibCopied ? "Copied!" : "Copy BibTeX"}
        </button>
        <span
          style={{
            fontSize: "0.9rem",
            color: bibCopied
              ? "var(--ifm-color-success)"
              : "var(--ifm-color-emphasis-600)",
          }}
        >
          {bibCopied
            ? "BibTeX copied to clipboard. Switch to Zotero and choose Import from Clipboard."
            : "Click to copy BibTeX to clipboard for Zotero import."}
        </span>
      </div>
      <pre style={codeBlockStyle}>
        <code>{bibtexCitation}</code>
      </pre>
    </div>
  );
}
