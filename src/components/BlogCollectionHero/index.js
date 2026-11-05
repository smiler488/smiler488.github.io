import React from "react";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import { useIsChinese } from "@site/src/components/ds";
import styles from "./styles.module.css";

const DEFAULTS = {
  en: {
    eyebrow: "Research notebook",
    title: "Research notes for measurable, reproducible science.",
    description:
      "Practical field notes at the intersection of artificial intelligence, plant phenotyping, imaging, and scientific software.",
    shortcuts: "Blog shortcuts",
    latest: "Browse latest notes",
    archive: "Explore archive",
    topics: "Browse topics",
  },
  zh: {
    eyebrow: "研究笔记本",
    title: "为可度量、可复现的科学而写的研究笔记。",
    description: "人工智能、植物表型、成像与科研软件交汇处的实用笔记。",
    shortcuts: "笔记快捷入口",
    latest: "浏览最新笔记",
    archive: "查看归档",
    topics: "按主题浏览",
  },
};

export default function BlogCollectionHero({
  eyebrow,
  title,
  description,
  stats = [],
  compact = false,
}) {
  const t = useIsChinese() ? DEFAULTS.zh : DEFAULTS.en;
  return (
    <header className={`${styles.hero} ${compact ? styles.compact : ""}`}>
      <div className={styles.content}>
        <p className={styles.eyebrow}>
          {eyebrow ?? t.eyebrow}
        </p>
        <Heading as="h1">{title ?? t.title}</Heading>
        <p className={styles.description}>{description ?? t.description}</p>

        {!compact && (
          <div className={styles.actions} aria-label={t.shortcuts}>
            <Link className={styles.primaryAction} to="#latest-notes">
              {t.latest}
            </Link>
            <Link className={styles.secondaryAction} to="/blog/archive">
              {t.archive}
            </Link>
            <Link className={styles.secondaryAction} to="/blog/tags">
              {t.topics}
            </Link>
          </div>
        )}

        {stats.length > 0 && (
          <dl className={styles.stats}>
            {stats.map((stat) => (
              <div key={`${stat.value}-${stat.label}`}>
                <dt>{stat.value}</dt>
                <dd>{stat.label}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
    </header>
  );
}
