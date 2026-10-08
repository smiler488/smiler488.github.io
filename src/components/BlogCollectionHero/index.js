import React from "react";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

const defaultStats = [
  { value: "19", label: "field notes" },
  {
    value: `2021–${new Date().getUTCFullYear()}`,
    label: "research timeline",
  },
  { value: "5", label: "focus areas" },
];

export default function BlogCollectionHero({
  eyebrow = "Research notebook",
  title = "Research notes for measurable, reproducible science.",
  description = "Practical field notes at the intersection of artificial intelligence, plant phenotyping, imaging, and scientific software.",
  stats = defaultStats,
  compact = false,
}) {
  return (
    <header className={`${styles.hero} ${compact ? styles.compact : ""}`}>
      <div className={styles.content}>
        <p className={styles.eyebrow}>
          {eyebrow}
        </p>
        <Heading as="h1">{title}</Heading>
        <p className={styles.description}>{description}</p>

        {!compact && (
          <div className={styles.actions} aria-label="Blog shortcuts">
            <Link className={styles.primaryAction} to="#latest-notes">
              Browse latest notes
            </Link>
            <Link className={styles.secondaryAction} to="/blog/archive">
              Explore archive
            </Link>
            <Link className={styles.secondaryAction} to="/blog/tags">
              Browse topics
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
