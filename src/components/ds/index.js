/**
 * Design-system primitives (design/DESIGN_SPEC.md §5.7). Every component
 * supports en / zh, light / dark and phone widths, and only uses semantic
 * tokens from src/css/tokens.css.
 */
import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import Head from "@docusaurus/Head";
import Heading from "@theme/Heading";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {
  LAYERS,
  getLayer,
  getPublication,
  doiUrl,
  formatApa,
  formatBibtex,
} from "@site/src/data/publications";
import { cvContent, cvIdentity } from "@site/src/data/cvData";
import styles from "./styles.module.css";

/** True on the zh-Hans locale. */
export function useIsChinese() {
  const { i18n } = useDocusaurusContext();
  return i18n.currentLocale === "zh-Hans";
}

/** Pick the current locale's string from a { en, zh } value. */
export function useLocalize() {
  const isChinese = useIsChinese();
  return (value) =>
    value && typeof value === "object" && "en" in value
      ? isChinese
        ? value.zh ?? value.en
        : value.en
      : value;
}

/** Inject one schema.org JSON-LD object into <head> (DESIGN_SPEC §9.1). */
export function JsonLd({ data }) {
  return (
    <Head>
      <script type="application/ld+json">{JSON.stringify(data)}</script>
    </Head>
  );
}

/** Unboxed page header: eyebrow, title, lead and optional stats/actions. */
export function PageHero({ eyebrow, title, lead, children, compact = false }) {
  return (
    <header
      className={clsx(styles.pageHero, compact && styles.pageHeroCompact)}
    >
      {eyebrow && <p className={styles.eyebrow}>{eyebrow}</p>}
      <Heading as="h1" className={styles.pageTitle}>
        {title}
      </Heading>
      {lead && <p className={styles.lead}>{lead}</p>}
      {children}
    </header>
  );
}

/** Section eyebrow + heading + optional description. */
export function SectionHeader({ id, eyebrow, title, description, as = "h2" }) {
  return (
    <div className={styles.sectionHeader}>
      <div>
        {eyebrow && <p className={styles.eyebrow}>{eyebrow}</p>}
        <Heading as={as} id={id} className={styles.sectionTitle}>
          {title}
        </Heading>
      </div>
      {description && (
        <p className={styles.sectionDescription}>{description}</p>
      )}
    </div>
  );
}

/** Row of plain statistics. items: [{ value, label }] */
export function Stats({ items }) {
  return (
    <dl className={styles.stats}>
      {items.map((item) => (
        <div key={item.label}>
          <dt>{item.value}</dt>
          <dd>{item.label}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Pill used for tags and filters. Renders a button when onClick is given. */
export function Chip({ children, to, onClick, active = false }) {
  const className = clsx(styles.chip, active && styles.chipActive);
  if (onClick) {
    return (
      <button
        type="button"
        className={className}
        aria-pressed={active}
        onClick={onClick}
      >
        {children}
      </button>
    );
  }
  if (to) {
    return (
      <Link className={className} to={to}>
        {children}
      </Link>
    );
  }
  return <span className={className}>{children}</span>;
}

/** The site's one notice pattern. tone: info | success | warning | danger */
export function Notice({ tone = "info", title, children }) {
  return (
    <div className={clsx(styles.notice, styles[`notice_${tone}`])} role="note">
      {title && <strong className={styles.noticeTitle}>{title}</strong>}
      <div>{children}</div>
    </div>
  );
}

/** Four-layer badge, e.g. "II · Understand". Links to the research hub filter. */
export function LayerBadge({ id, link = true }) {
  const localize = useLocalize();
  const layer = getLayer(id);
  if (!layer) return null;
  const label = (
    <>
      <span className={styles.layerIndex}>{layer.index}</span>
      {localize(layer.name)}
    </>
  );
  return link ? (
    <Link className={styles.layerBadge} to={`/research?layer=${layer.id}`}>
      {label}
    </Link>
  ) : (
    <span className={styles.layerBadge}>{label}</span>
  );
}

export function LayerBadges({ ids = [], link = true }) {
  if (!ids.length) return null;
  return (
    <span className={styles.layerBadges}>
      {ids.map((id) => (
        <LayerBadge key={id} id={id} link={link} />
      ))}
    </span>
  );
}

export { LAYERS };

function CopyButton({ text, label, copiedLabel }) {
  const [copied, setCopied] = React.useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }
  return (
    <button type="button" className={styles.evidenceLink} onClick={copy}>
      {copied ? copiedLabel : label}
    </button>
  );
}

const EVIDENCE_COPY = {
  en: {
    label: "Evidence",
    paper: "Paper",
    software: "Software DOI",
    code: "Code",
    data: "Data",
    tryIt: "Try it in the Lab",
    bibtex: "Copy BibTeX",
    apa: "Copy citation",
    copied: "Copied",
  },
  zh: {
    label: "证据",
    paper: "论文",
    software: "软件 DOI",
    code: "代码",
    data: "数据",
    tryIt: "在实验室中试用",
    bibtex: "复制 BibTeX",
    apa: "复制引用",
    copied: "已复制",
  },
};

/**
 * One row of evidence entry points: paper DOI, code, data, the related tool,
 * and citation copy buttons. Only links that exist are rendered.
 */
export function EvidenceBar({ publications = [], code, data, tool }) {
  const isChinese = useIsChinese();
  const copy = isChinese ? EVIDENCE_COPY.zh : EVIDENCE_COPY.en;
  const pubs = publications.map(getPublication).filter(Boolean);
  const primary = pubs[0];
  const links = [
    ...pubs.map((pub) => ({
      key: pub.id,
      label: pub.type === "software" ? copy.software : copy.paper,
      href: doiUrl(pub.doi),
    })),
    code && { key: "code", label: copy.code, href: code },
    data && { key: "data", label: copy.data, href: data },
  ].filter(Boolean);

  return (
    <nav className={styles.evidenceBar} aria-label={copy.label}>
      {tool && (
        <Link className={styles.evidencePrimary} to={tool}>
          {copy.tryIt}
        </Link>
      )}
      {links.map((link) => (
        <Link key={link.key} className={styles.evidenceLink} href={link.href}>
          {link.label}
        </Link>
      ))}
      {primary && (
        <>
          <CopyButton
            text={formatApa(primary)}
            label={copy.apa}
            copiedLabel={copy.copied}
          />
          <CopyButton
            text={formatBibtex(primary)}
            label={copy.bibtex}
            copiedLabel={copy.copied}
          />
        </>
      )}
    </nav>
  );
}

const TOOLBOX = [
  {
    to: "/resources",
    title: { en: "Learning resources", zh: "学习资源" },
    hint: {
      en: "Curated courses, models and datasets",
      zh: "精选课程、模型与数据集",
    },
  },
  {
    to: "/navigator",
    title: { en: "Navigator", zh: "网址导航" },
    hint: { en: "A directory of research links", zh: "科研常用网址目录" },
  },
  {
    to: "/mpicks",
    title: { en: "mPicks", zh: "好物推荐" },
    hint: {
      en: "Tools worth adding to your setup",
      zh: "值得纳入工作流的好物",
    },
  },
];

/**
 * Entry points to the utility pages that live outside the primary
 * navigation (DESIGN_SPEC §4.1). `extra` prepends page-specific links.
 */
export function ToolboxLinks({ extra = [] }) {
  const isChinese = useIsChinese();
  const localize = useLocalize();
  const items = [...extra, ...TOOLBOX];
  return (
    <nav
      className={styles.toolbox}
      aria-label={isChinese ? "工具箱" : "Toolbox"}
    >
      <p className={styles.toolboxLabel}>{isChinese ? "工具箱" : "Toolbox"}</p>
      <ul className={styles.toolboxList}>
        {items.map((item) => (
          <li key={item.to}>
            <Link to={item.to}>
              <strong>{localize(item.title)}</strong>
              <span>{localize(item.hint)}</span>
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/** schema.org Person for the site owner, from the CV data (DESIGN_SPEC §9.1). */
export function PersonJsonLd() {
  const isChinese = useIsChinese();
  const { siteConfig } = useDocusaurusContext();
  const hero = cvContent[isChinese ? "zh" : "en"].hero;
  const institution = cvContent.en.hero.institution.split(" · ")[0];
  return (
    <JsonLd
      data={{
        "@context": "https://schema.org",
        "@type": "Person",
        name: "Liangchao Deng",
        alternateName: "邓良超",
        jobTitle: hero.role,
        affiliation: { "@type": "Organization", name: institution },
        alumniOf: {
          "@type": "CollegeOrUniversity",
          name: "Shihezi University",
        },
        url: siteConfig.url,
        email: `mailto:${cvIdentity.academicEmail}`,
        sameAs: [cvIdentity.orcid, cvIdentity.scholar, cvIdentity.github],
        knowsAbout: [
          "Plant phenotyping",
          "Crop modeling",
          "Computer vision",
          "Remote sensing",
          "3D reconstruction",
          "Canopy photosynthesis",
        ],
      }}
    />
  );
}
