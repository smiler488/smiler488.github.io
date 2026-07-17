import React from "react";
import clsx from "clsx";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { getAppById, localizeApp } from "../../data/appManifest";
import styles from "./styles.module.css";

const FALLBACK = {
  en: {
    title: "Research tool",
    eyebrow: "Browser research tool",
    runtime: "Browser research workspace",
    back: "App Lab",
    nav: "App Lab navigation",
    capabilities: "Tool capabilities",
    guide: "Guide",
  },
  zh: {
    title: "科研工具",
    eyebrow: "浏览器科研工具",
    runtime: "浏览器科研工作台",
    back: "应用实验室",
    nav: "应用实验室导航",
    capabilities: "工具能力",
    guide: "教程",
  },
};

export default function AppScaffold({
  appId,
  title: titleProp,
  eyebrow: eyebrowProp,
  description: descriptionProp,
  icon: iconProp,
  tone: toneProp,
  badges: badgesProp,
  tutorialHref: tutorialHrefProp,
  actions,
  children,
  className,
}) {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const t = isChinese ? FALLBACK.zh : FALLBACK.en;

  const app = appId ? getAppById(appId) : null;
  const title = titleProp || localizeApp(app?.name, isChinese) || t.title;
  const eyebrow =
    eyebrowProp || localizeApp(app?.categoryLabel, isChinese) || t.eyebrow;
  const description =
    descriptionProp || localizeApp(app?.description, isChinese);
  const icon = iconProp || app?.icon || "LAB";
  const tone = toneProp || app?.tone || "blue";
  const badges = badgesProp || localizeApp(app?.badges, isChinese) || [];
  const tutorialHref = tutorialHrefProp || app?.tutorial;
  const runtime = localizeApp(app?.runtime, isChinese) || t.runtime;
  // Derive the id from the stable app id — a localized title would collapse
  // into a string of dashes once it is no longer Latin script.
  const titleId = `app-title-${appId || "tool"}`;

  return (
    <Layout title={title} description={description}>
      <main className={styles.page} data-tone={tone} aria-labelledby={titleId}>
        <div className={styles.ambient} aria-hidden="true">
          <span className={styles.ambientOrb} />
          <span className={styles.ambientGrid} />
        </div>

        <div className={styles.container}>
          <nav className={styles.contextBar} aria-label={t.nav}>
            <Link className={styles.backLink} to="/app">
              <span aria-hidden="true">←</span>
              <span>{t.back}</span>
            </Link>
            <span className={styles.contextMeta}>
              <span className={styles.statusDot} aria-hidden="true" />
              {runtime}
            </span>
          </nav>

          <header className={styles.hero}>
            <div className={styles.heroCopy}>
              <div className={styles.iconTile} aria-hidden="true">
                {icon}
              </div>
              <div>
                <p className={styles.eyebrow}>{eyebrow}</p>
                <Heading as="h1" className={styles.title} id={titleId}>
                  {title}
                </Heading>
                {description && (
                  <p className={styles.description}>{description}</p>
                )}
                {badges.length > 0 && (
                  <ul className={styles.badges} aria-label={t.capabilities}>
                    {badges.map((badge) => (
                      <li key={badge}>{badge}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {(tutorialHref || actions) && (
              <div className={styles.heroActions}>
                {tutorialHref && (
                  <Link className={styles.guideLink} to={tutorialHref}>
                    {t.guide} <span aria-hidden="true">↗</span>
                  </Link>
                )}
                {actions}
              </div>
            )}
          </header>

          <div className={clsx(styles.workspace, className)}>{children}</div>
        </div>
      </main>
    </Layout>
  );
}
