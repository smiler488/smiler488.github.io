import React from "react";
import clsx from "clsx";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import Link from "@docusaurus/Link";
import { getAppById } from "../../data/appManifest";
import styles from "./styles.module.css";

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
  const app = appId ? getAppById(appId) : null;
  const title = titleProp || app?.name || "Research tool";
  const eyebrow = eyebrowProp || app?.categoryLabel || "Browser research tool";
  const description = descriptionProp || app?.description;
  const icon = iconProp || app?.icon || "LAB";
  const tone = toneProp || app?.tone || "blue";
  const badges = badgesProp || app?.badges || [];
  const tutorialHref = tutorialHrefProp || app?.tutorial;
  const runtime = app?.runtime || "Browser research workspace";
  const titleId = `app-title-${title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")}`;

  return (
    <Layout title={title} description={description}>
      <main className={styles.page} data-tone={tone} aria-labelledby={titleId}>
        <div className={styles.ambient} aria-hidden="true">
          <span className={styles.ambientOrb} />
          <span className={styles.ambientGrid} />
        </div>

        <div className={styles.container}>
          <nav className={styles.contextBar} aria-label="App Lab navigation">
            <Link className={styles.backLink} to="/app">
              <span aria-hidden="true">←</span>
              <span>App Lab</span>
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
                  <ul className={styles.badges} aria-label="Tool capabilities">
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
                    Guide <span aria-hidden="true">↗</span>
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
