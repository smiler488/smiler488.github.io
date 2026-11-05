import React from "react";
import clsx from "clsx";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { getAppById, localizeApp } from "../../data/appManifest";
import { JsonLd } from "../ds";
import WorkbenchPanels, { ToolStatus } from "../Workbench";
import styles from "./styles.module.css";

const FALLBACK = {
  en: {
    title: "Research tool",
    eyebrow: "Browser research tool",
    runtime: "Browser research workspace",
    back: "App Lab",
    nav: "App Lab navigation",
    capabilities: "Tool capabilities",
    guide: "Read the guide",
  },
  zh: {
    title: "科研工具",
    eyebrow: "浏览器科研工具",
    runtime: "浏览器科研工作台",
    back: "应用实验室",
    nav: "应用实验室导航",
    capabilities: "工具能力",
    guide: "使用教程",
  },
};

/** Book icon: closed at rest, open on hover and focus (muse icon rule). */
function GuideIcon() {
  const line = {
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.75,
    strokeLinecap: "round",
    strokeLinejoin: "round",
  };
  return (
    <span className={styles.guideIcon} aria-hidden="true">
      <svg className={styles.guideClosed} viewBox="0 0 24 24" {...line}>
        <path d="M5 4.5A1.5 1.5 0 0 1 6.5 3H19v15H6.5A1.5 1.5 0 0 0 5 19.5Z" />
        <path d="M5 19.5A1.5 1.5 0 0 0 6.5 21H19v-3" />
        <path d="M9 7h6" />
      </svg>
      <svg className={styles.guideOpen} viewBox="0 0 24 24" {...line}>
        <path d="M12 6.5C10.4 5 8 4.5 3 4.5v13c5 0 7.4.5 9 2 1.6-1.5 4-2 9-2v-13c-5 0-7.4.5-9 2Z" />
        <path d="M12 6.5v13" />
      </svg>
    </span>
  );
}

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
  const appUrl = useBaseUrl(app?.route ?? "/app", { absolute: true });

  return (
    <Layout title={title} description={description}>
      {app && (
        <JsonLd
          data={{
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            name: title,
            description,
            url: appUrl,
            applicationCategory: "UtilitiesApplication",
            applicationSubCategory: eyebrow,
            operatingSystem: "Web browser",
            isAccessibleForFree: true,
            offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
            author: { "@type": "Person", name: "Liangchao Deng" },
          }}
        />
      )}
      <main className={styles.page} data-tone={tone} aria-labelledby={titleId}>
        <div className={styles.container}>
          <nav className={styles.contextBar} aria-label={t.nav}>
            <Link className={styles.backLink} to="/app">
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
                <ToolStatus app={app} />
              </div>
            </div>

            {(tutorialHref || actions) && (
              <div className={styles.heroActions}>
                {tutorialHref && (
                  <Link className={styles.guideLink} to={tutorialHref}>
                    <GuideIcon />
                    <span>{t.guide}</span>
                  </Link>
                )}
                {actions}
              </div>
            )}
          </header>

          <div className={clsx(styles.workspace, className)}>{children}</div>
          <WorkbenchPanels app={app} />
        </div>
      </main>
    </Layout>
  );
}
