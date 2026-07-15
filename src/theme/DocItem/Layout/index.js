import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import { useWindowSize } from "@docusaurus/theme-common";
import { useDoc } from "@docusaurus/plugin-content-docs/client";
import Heading from "@theme/Heading";
import DocItemPaginator from "@theme/DocItem/Paginator";
import DocVersionBanner from "@theme/DocVersionBanner";
import DocVersionBadge from "@theme/DocVersionBadge";
import DocItemFooter from "@theme/DocItem/Footer";
import DocItemTOCMobile from "@theme/DocItem/TOC/Mobile";
import DocItemTOCDesktop from "@theme/DocItem/TOC/Desktop";
import DocItemContent from "@theme/DocItem/Content";
import DocBreadcrumbs from "@theme/DocBreadcrumbs";
import ContentVisibility from "@theme/ContentVisibility";
import Comment from "@site/src/components/comment";
import styles from "./styles.module.css";

/** Decide if the table of contents should render on this viewport. */
function useDocTOC() {
  const { frontMatter, toc } = useDoc();
  const windowSize = useWindowSize();
  const hidden = frontMatter.hide_table_of_contents;
  const canRender = !hidden && toc.length > 0;
  const mobile = canRender ? <DocItemTOCMobile /> : undefined;
  const desktop =
    canRender && (windowSize === "desktop" || windowSize === "ssr") ? (
      <DocItemTOCDesktop />
    ) : undefined;

  return { hidden, mobile, desktop };
}

function AppTutorialHero({ metadata, frontMatter }) {
  const badges = Array.isArray(frontMatter.app_badges)
    ? frontMatter.app_badges
    : [];
  const tone = frontMatter.app_tone || "blue";

  return (
    <header className={styles.tutorialHero}>
      <div className={styles.tutorialHeroTopline}>
        <Link
          className={styles.tutorialBackLink}
          to="/docs/category/tutorial---apps"
        >
          <span aria-hidden="true">←</span> All App Lab guides
        </Link>
        {frontMatter.app_runtime && (
          <span className={styles.tutorialRuntime}>
            <span className={styles.runtimeDot} aria-hidden="true" />
            {frontMatter.app_runtime}
          </span>
        )}
      </div>

      <div className={styles.tutorialHeroBody}>
        <div
          className={styles.tutorialIcon}
          data-tone={tone}
          aria-hidden="true"
        >
          {frontMatter.app_icon || "APP"}
        </div>
        <div className={styles.tutorialHeroCopy}>
          <span className={styles.tutorialEyebrow}>
            {frontMatter.app_category || "App Lab"} · practical guide
          </span>
          <Heading as="h1">{metadata.title}</Heading>
          <p>{metadata.description}</p>
        </div>
      </div>

      <div className={styles.tutorialHeroFooter}>
        {badges.length > 0 && (
          <ul className={styles.tutorialBadges} aria-label="Key capabilities">
            {badges.map((badge) => (
              <li key={badge}>{badge}</li>
            ))}
          </ul>
        )}
        <Link className={styles.openAppButton} to={frontMatter.app_route}>
          Open tool <span aria-hidden="true">↗</span>
        </Link>
      </div>
    </header>
  );
}

function StandardDocLayout({ children, docTOC, metadata }) {
  return (
    <div className="row">
      <div className={clsx("col", !docTOC.hidden && styles.docItemCol)}>
        <ContentVisibility metadata={metadata} />
        <DocVersionBanner />
        <div className={styles.docItemContainer}>
          <article>
            <DocBreadcrumbs />
            <DocVersionBadge />
            {docTOC.mobile}
            <DocItemContent>{children}</DocItemContent>
            <DocItemFooter />
          </article>
          <DocItemPaginator />
        </div>
        <Comment />
      </div>
      {docTOC.desktop && <div className="col col--3">{docTOC.desktop}</div>}
    </div>
  );
}

function AppTutorialLayout({ children, docTOC, metadata, frontMatter }) {
  return (
    <div className={styles.tutorialPage}>
      <span
        className={clsx(styles.tutorialGlow, styles.tutorialGlowOne)}
        aria-hidden="true"
      />
      <span
        className={clsx(styles.tutorialGlow, styles.tutorialGlowTwo)}
        aria-hidden="true"
      />
      <div className="row">
        <div
          className={clsx("col", !docTOC.hidden && styles.tutorialDocItemCol)}
        >
          <ContentVisibility metadata={metadata} />
          <DocVersionBanner />
          <div className={styles.tutorialBreadcrumbs}>
            <DocBreadcrumbs />
            <DocVersionBadge />
          </div>
          <AppTutorialHero metadata={metadata} frontMatter={frontMatter} />
          <div className={styles.tutorialArticle}>
            <article>
              {docTOC.mobile}
              <DocItemContent>{children}</DocItemContent>
              <DocItemFooter />
            </article>
            <DocItemPaginator />
          </div>
          <Comment />
        </div>
        {docTOC.desktop && (
          <aside className={clsx("col col--3", styles.tutorialToc)}>
            {docTOC.desktop}
          </aside>
        )}
      </div>
    </div>
  );
}

export default function DocItemLayout({ children }) {
  const docTOC = useDocTOC();
  const { metadata, frontMatter } = useDoc();
  const isAppTutorial =
    typeof frontMatter.app_route === "string" &&
    frontMatter.app_route.startsWith("/app/");

  if (isAppTutorial) {
    return (
      <AppTutorialLayout
        docTOC={docTOC}
        metadata={metadata}
        frontMatter={frontMatter}
      >
        {children}
      </AppTutorialLayout>
    );
  }

  return (
    <StandardDocLayout docTOC={docTOC} metadata={metadata}>
      {children}
    </StandardDocLayout>
  );
}
