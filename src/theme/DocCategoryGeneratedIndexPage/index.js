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
import { APP_CATEGORIES, APP_MANIFEST } from "@site/src/data/appManifest";
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

function TutorialCard({ app }) {
  return (
    <article className={styles.tutorialCard} data-tone={app.tone}>
      <div className={styles.cardTopline}>
        <span className={styles.cardIcon} aria-hidden="true">
          {app.icon}
        </span>
        <span className={styles.cardCategory}>{app.categoryLabel}</span>
      </div>
      <div className={styles.cardCopy}>
        <Heading as="h3">
          <Link to={app.tutorial}>{app.name}</Link>
        </Heading>
        <p>{app.description}</p>
      </div>
      <ul className={styles.cardBadges} aria-label={`${app.name} capabilities`}>
        {app.badges.slice(0, 3).map((badge) => (
          <li key={badge}>{badge}</li>
        ))}
      </ul>
      <div className={styles.cardFooter}>
        <span className={styles.cardRuntime}>{app.runtime}</span>
        <div className={styles.cardLinks}>
          <Link
            className={styles.launchLink}
            to={app.route}
            aria-label={`Open ${app.name}`}
          >
            Open <span aria-hidden="true">↗</span>
          </Link>
          <Link className={styles.guideLink} to={app.tutorial}>
            Read guide <span aria-hidden="true">→</span>
          </Link>
        </div>
      </div>
    </article>
  );
}

function AppTutorialCategoryPage({ categoryGeneratedIndex }) {
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
          <span className={styles.eyebrow}>APP LAB DOCUMENTATION</span>
          <Heading as="h1">
            A practical guide for every App Lab workflow.
          </Heading>
          <p>
            Learn what each tool really does, what data leaves your browser, and
            where scientific or engineering validation is still required. The
            names and descriptions below come from the same manifest as App Lab.
          </p>
          <div className={styles.heroActions}>
            <Link className={styles.primaryAction} to="/app">
              Open App Lab <span aria-hidden="true">↗</span>
            </Link>
            <Link className={styles.secondaryAction} to="#app-guides">
              Browse 14 guides <span aria-hidden="true">↓</span>
            </Link>
          </div>
        </div>
        <dl
          className={styles.heroStats}
          aria-label="Tutorial collection summary"
        >
          <div>
            <dt>14</dt>
            <dd>tool guides</dd>
          </div>
          <div>
            <dt>5</dt>
            <dd>workflow areas</dd>
          </div>
          <div>
            <dt>1</dt>
            <dd>shared source of truth</dd>
          </div>
        </dl>
      </header>

      <main className={styles.guideDirectory}>
        <div className={styles.directoryIntro}>
          <div>
            <span className={styles.sectionLabel}>GUIDE DIRECTORY</span>
            <Heading as="h2" id="app-guides">
              Choose a workflow
            </Heading>
          </div>
          <p>
            Each guide includes a verified quick start, controls and outputs,
            privacy boundaries, limitations, and troubleshooting steps.
          </p>
        </div>

        {groups.map((group) => (
          <section
            className={styles.guideGroup}
            key={group.id}
            aria-labelledby={`guide-group-${group.id}`}
          >
            <div className={styles.groupHeading}>
              <Heading as="h2" id={`guide-group-${group.id}`}>
                {group.label}
              </Heading>
              <span>
                {group.apps.length}{" "}
                {group.apps.length === 1 ? "guide" : "guides"}
              </span>
            </div>
            <div className={styles.tutorialGrid}>
              {group.apps.map((app) => (
                <TutorialCard app={app} key={app.id} />
              ))}
            </div>
          </section>
        ))}
      </main>

      <section className={styles.collectionNote}>
        <div>
          <span className={styles.sectionLabel}>KEEP THE CONTEXT</span>
          <Heading as="h2">Use the guide and the tool side by side.</Heading>
        </div>
        <p>
          App Lab tools are browser-first research utilities, not substitutes
          for calibrated instruments, official databases, or professional
          engineering review. Every guide calls out those boundaries explicitly.
        </p>
        <Link className={styles.secondaryAction} to="/app">
          Return to App Lab <span aria-hidden="true">→</span>
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
