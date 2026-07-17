import React from "react";
import Link from "@docusaurus/Link";
import { PageMetadata } from "@docusaurus/theme-common";
import { useCurrentSidebarCategory } from "@docusaurus/plugin-content-docs/client";
import useBaseUrl from "@docusaurus/useBaseUrl";
import DocCardList from "@theme/DocCardList";
import DocPaginator from "@theme/DocPaginator";
import DocVersionBanner from "@theme/DocVersionBanner";
import DocVersionBadge from "@theme/DocVersionBadge";
import DocBreadcrumbs from "@theme/DocBreadcrumbs";
import Heading from "@theme/Heading";
import {
  APP_CATEGORIES,
  APP_MANIFEST,
  localizeApp,
} from "@site/src/data/appManifest";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./styles.module.css";

function CategoryMetadata({ categoryGeneratedIndex }) {
  return (
    <PageMetadata
      title={categoryGeneratedIndex.title}
      description={categoryGeneratedIndex.description}
      keywords={categoryGeneratedIndex.keywords}
      image={useBaseUrl(categoryGeneratedIndex.image)}
    />
  );
}

function CategoryFooter({ categoryGeneratedIndex }) {
  return (
    <footer className={styles.categoryFooter}>
      <DocPaginator
        previous={categoryGeneratedIndex.navigation.previous}
        next={categoryGeneratedIndex.navigation.next}
      />
    </footer>
  );
}

function DefaultCategoryPage({ categoryGeneratedIndex }) {
  const category = useCurrentSidebarCategory();

  return (
    <div className={styles.generatedIndexPage}>
      <DocVersionBanner />
      <DocBreadcrumbs />
      <DocVersionBadge />
      <header>
        <Heading as="h1" className={styles.title}>
          {categoryGeneratedIndex.title}
        </Heading>
        {categoryGeneratedIndex.description && (
          <p>{categoryGeneratedIndex.description}</p>
        )}
      </header>
      <article className="margin-top--lg">
        <DocCardList items={category.items} className={styles.list} />
      </article>
      <CategoryFooter categoryGeneratedIndex={categoryGeneratedIndex} />
    </div>
  );
}

const GUIDE_COPY = {
  en: {
    eyebrow: "APP LAB DOCUMENTATION",
    title: "A practical guide for every App Lab workflow.",
    intro:
      "Learn what each tool really does, what data leaves your browser, and where scientific or engineering validation is still required. The names and descriptions below come from the same manifest as App Lab.",
    openLab: "Open App Lab",
    browse: (n) => `Browse ${n} guides`,
    statsLabel: "Tutorial collection summary",
    statGuides: "tool guides",
    statAreas: "workflow areas",
    statSource: "shared source of truth",
    directoryLabel: "GUIDE DIRECTORY",
    directoryTitle: "Choose a workflow",
    directoryText:
      "Each guide includes a verified quick start, controls and outputs, privacy boundaries, limitations, and troubleshooting steps.",
    guide: "guide",
    guides: "guides",
    noteLabel: "KEEP THE CONTEXT",
    noteTitle: "Use the guide and the tool side by side.",
    noteText:
      "App Lab tools are browser-first research utilities, not substitutes for calibrated instruments, official databases, or professional engineering review. Every guide calls out those boundaries explicitly.",
    back: "Return to App Lab",
    open: "Open",
    readGuide: "Read guide",
    capabilities: (n) => `${n} capabilities`,
    openAria: (n) => `Open ${n}`,
  },
  zh: {
    eyebrow: "应用实验室文档",
    title: "为每一个应用实验室工作流提供的实用指南。",
    intro:
      "了解每个工具的真实功能、哪些数据会离开你的浏览器，以及哪些环节仍需科学或工程验证。下方的名称与描述与应用实验室来自同一份清单。",
    openLab: "进入应用实验室",
    browse: (n) => `浏览 ${n} 篇指南`,
    statsLabel: "教程合集概览",
    statGuides: "篇工具指南",
    statAreas: "个工作流领域",
    statSource: "份统一数据源",
    directoryLabel: "指南目录",
    directoryTitle: "选择一个工作流",
    directoryText:
      "每篇指南都包含经过验证的快速上手、控件与输出、隐私边界、局限性与排障步骤。",
    guide: "篇指南",
    guides: "篇指南",
    noteLabel: "别丢掉上下文",
    noteTitle: "指南与工具请对照使用。",
    noteText:
      "应用实验室的工具是浏览器优先的科研辅助工具，不能替代经过标定的仪器、官方数据库或专业工程审查。每篇指南都会明确指出这些边界。",
    back: "返回应用实验室",
    open: "打开",
    readGuide: "阅读指南",
    capabilities: (n) => `${n} 功能`,
    openAria: (n) => `打开 ${n}`,
  },
};

function TutorialCard({ app, isChinese, copy }) {
  const name = localizeApp(app.name, isChinese);
  const badges = localizeApp(app.badges, isChinese);
  return (
    <article className={styles.tutorialCard} data-tone={app.tone}>
      <div className={styles.cardTopline}>
        <span className={styles.cardIcon} aria-hidden="true">
          {app.icon}
        </span>
        <span className={styles.cardCategory}>
          {localizeApp(app.categoryLabel, isChinese)}
        </span>
      </div>
      <div className={styles.cardCopy}>
        <Heading as="h3">
          <Link to={app.tutorial}>{name}</Link>
        </Heading>
        <p>{localizeApp(app.description, isChinese)}</p>
      </div>
      <ul className={styles.cardBadges} aria-label={copy.capabilities(name)}>
        {badges.slice(0, 3).map((badge) => (
          <li key={badge}>{badge}</li>
        ))}
      </ul>
      <div className={styles.cardFooter}>
        <span className={styles.cardRuntime}>
          {localizeApp(app.runtime, isChinese)}
        </span>
        <div className={styles.cardLinks}>
          <Link
            className={styles.launchLink}
            to={app.route}
            aria-label={copy.openAria(name)}
          >
            {copy.open} <span aria-hidden="true">↗</span>
          </Link>
          <Link className={styles.guideLink} to={app.tutorial}>
            {copy.readGuide} <span aria-hidden="true">→</span>
          </Link>
        </div>
      </div>
    </article>
  );
}

function AppTutorialCategoryPage({ categoryGeneratedIndex }) {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? GUIDE_COPY.zh : GUIDE_COPY.en;
  const groups = APP_CATEGORIES.filter(({ id }) => id !== "all").map(
    (category) => ({
      ...category,
      apps: APP_MANIFEST.filter((app) => app.category === category.id),
    })
  );

  return (
    <div className={styles.tutorialIndex}>
      <span className={styles.ambientOrbOne} aria-hidden="true" />
      <span className={styles.ambientOrbTwo} aria-hidden="true" />
      <DocVersionBanner />
      <div className={styles.breadcrumbRow}>
        <DocBreadcrumbs />
        <DocVersionBadge />
      </div>

      <header className={styles.tutorialHero}>
        <div className={styles.heroCopy}>
          <span className={styles.eyebrow}>{copy.eyebrow}</span>
          <Heading as="h1">{copy.title}</Heading>
          <p>{copy.intro}</p>
          <div className={styles.heroActions}>
            <Link className={styles.primaryAction} to="/app">
              {copy.openLab} <span aria-hidden="true">↗</span>
            </Link>
            <Link className={styles.secondaryAction} to="#app-guides">
              {copy.browse(APP_MANIFEST.length)}{" "}
              <span aria-hidden="true">↓</span>
            </Link>
          </div>
        </div>
        <dl className={styles.heroStats} aria-label={copy.statsLabel}>
          <div>
            <dt>{APP_MANIFEST.length}</dt>
            <dd>{copy.statGuides}</dd>
          </div>
          <div>
            <dt>{APP_CATEGORIES.length - 1}</dt>
            <dd>{copy.statAreas}</dd>
          </div>
          <div>
            <dt>1</dt>
            <dd>{copy.statSource}</dd>
          </div>
        </dl>
      </header>

      <main className={styles.guideDirectory}>
        <div className={styles.directoryIntro}>
          <div>
            <span className={styles.sectionLabel}>{copy.directoryLabel}</span>
            <Heading as="h2" id="app-guides">
              {copy.directoryTitle}
            </Heading>
          </div>
          <p>{copy.directoryText}</p>
        </div>

        {groups.map((group) => (
          <section
            className={styles.guideGroup}
            key={group.id}
            aria-labelledby={`guide-group-${group.id}`}
          >
            <div className={styles.groupHeading}>
              <Heading as="h2" id={`guide-group-${group.id}`}>
                {localizeApp(group.label, isChinese)}
              </Heading>
              <span>
                {group.apps.length}{" "}
                {group.apps.length === 1 ? copy.guide : copy.guides}
              </span>
            </div>
            <div className={styles.tutorialGrid}>
              {group.apps.map((app) => (
                <TutorialCard
                  app={app}
                  key={app.id}
                  isChinese={isChinese}
                  copy={copy}
                />
              ))}
            </div>
          </section>
        ))}
      </main>

      <section className={styles.collectionNote}>
        <div>
          <span className={styles.sectionLabel}>{copy.noteLabel}</span>
          <Heading as="h2">{copy.noteTitle}</Heading>
        </div>
        <p>{copy.noteText}</p>
        <Link className={styles.secondaryAction} to="/app">
          {copy.back} <span aria-hidden="true">→</span>
        </Link>
      </section>

      <CategoryFooter categoryGeneratedIndex={categoryGeneratedIndex} />
    </div>
  );
}

export default function DocCategoryGeneratedIndexPage(props) {
  const isAppTutorialCategory = props.categoryGeneratedIndex.permalink.includes(
    "/category/tutorial---apps"
  );

  return (
    <>
      <CategoryMetadata {...props} />
      {isAppTutorialCategory ? (
        <AppTutorialCategoryPage {...props} />
      ) : (
        <DefaultCategoryPage {...props} />
      )}
    </>
  );
}
