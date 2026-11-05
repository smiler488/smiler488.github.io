import React from "react";
import clsx from "clsx";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {
  HtmlClassNameProvider,
  PageMetadata,
  ThemeClassNames,
} from "@docusaurus/theme-common";
import BlogCollectionHero from "@site/src/components/BlogCollectionHero";
import BlogListPaginator from "@theme/BlogListPaginator";
import BlogListPageStructuredData from "@theme/BlogListPage/StructuredData";
import BlogPostItems from "@theme/BlogPostItems";
import Heading from "@theme/Heading";
import Layout from "@theme/Layout";
import SearchMetadata from "@theme/SearchMetadata";
import { useHistory, useLocation } from "@docusaurus/router";
import { usePluginData } from "@docusaurus/useGlobalData";
import NoteCard from "@site/src/components/NoteCard";
import { Chip, useIsChinese } from "@site/src/components/ds";
import styles from "./styles.module.css";

function BlogListPageMetadata({ metadata }) {
  const {
    siteConfig: { title: siteTitle },
  } = useDocusaurusContext();
  const { blogDescription, blogTitle, permalink } = metadata;
  const title = permalink === "/" ? siteTitle : blogTitle;

  return (
    <>
      <PageMetadata title={title} description={blogDescription} />
      <SearchMetadata tag="blog_posts_list" />
    </>
  );
}

const COPY = {
  en: {
    notes: (n) => (n === 1 ? "field note" : "field notes"),
    timeline: "research timeline",
    pages: (n) => (n > 1 ? `${n} pages` : "one collection"),
    reading: "curated reading",
    kicker: "Latest notes",
    title: "Ideas, methods, and working systems",
    intro:
      "Concise project records and technical guides, revised to separate verified results from experimental workflows.",
    filterLabel: "Show",
    kinds: { all: "All", research: "Research notes", guides: "Guides" },
  },
  zh: {
    notes: () => "篇笔记",
    timeline: "研究时间线",
    pages: (n) => (n > 1 ? `${n} 页` : "一个合集"),
    reading: "精选阅读",
    kicker: "最新笔记",
    title: "想法、方法与可运行的系统",
    intro: "简明的项目记录与技术指南，区分已验证的结果与实验性的流程。",
    filterLabel: "显示",
    kinds: { all: "全部", research: "研究笔记", guides: "教程" },
  },
};

const KINDS = ["all", "research", "guides"];

/**
 * Which shelf a note belongs on: "research" (part of the four-layer program
 * shown on /research) or "guides" (tooling and workflow how-tos).
 * `note` is a site-index entry: { layers, articleType, category, tags, ... }.
 */
function noteKind(note) {
  // Same rule as the /research hub, so both pages count the same notes
  return note.layers?.length ? "research" : "guides";
}

/** ?kind=research|guides, shared by the chips and the /research link. */
function useKindFilter() {
  const location = useLocation();
  const history = useHistory();
  const param = new URLSearchParams(location.search).get("kind");
  const active = KINDS.includes(param) ? param : "all";
  const setActive = (kind) => {
    history.replace({
      ...location,
      search: kind === "all" ? "" : `?kind=${kind}`,
    });
  };
  return [active, setActive];
}

function BlogListPageContent({ metadata, items }) {
  const isChinese = useIsChinese();
  const copy = isChinese ? COPY.zh : COPY.en;
  const { notes = [] } = usePluginData("site-index") ?? {};
  const [kind, setKind] = useKindFilter();
  const counts = {
    all: notes.length,
    research: notes.filter((note) => noteKind(note) === "research").length,
    guides: notes.filter((note) => noteKind(note) === "guides").length,
  };
  // A filter spans every page, so it reads the full index instead of `items`
  const filtered =
    kind === "all"
      ? null
      : notes
          .filter((note) => noteKind(note) === kind)
          .sort((a, b) => new Date(b.date) - new Date(a.date));

  return (
    <Layout>
      <main className={styles.page}>
        <BlogCollectionHero
          stats={[
            {
              value: String(metadata.totalCount),
              label: copy.notes(metadata.totalCount),
            },
            {
              value: `2021–${new Date().getUTCFullYear()}`,
              label: copy.timeline,
            },
            { value: copy.pages(metadata.totalPages), label: copy.reading },
          ]}
        />

        <section className={styles.collection} aria-labelledby="latest-notes">
          <div className={styles.collectionHeader}>
            <div>
              <p className={styles.kicker}>{copy.kicker}</p>
              <Heading as="h2" id="latest-notes">
                {copy.title}
              </Heading>
            </div>
            <p>{copy.intro}</p>
          </div>

          <div
            className={styles.filters}
            role="group"
            aria-label={copy.filterLabel}
          >
            {KINDS.map((id) => (
              <Chip key={id} active={kind === id} onClick={() => setKind(id)}>
                {copy.kinds[id]}
                <span className={styles.filterCount}>{counts[id]}</span>
              </Chip>
            ))}
          </div>

          {filtered ? (
            <div className={styles.grid}>
              {filtered.map((note) => (
                <NoteCard key={note.permalink} {...note} />
              ))}
            </div>
          ) : (
            <>
              <div className={styles.grid}>
                <BlogPostItems items={items} />
              </div>
              <div className={styles.paginator}>
                <BlogListPaginator metadata={metadata} />
              </div>
            </>
          )}
        </section>
      </main>
    </Layout>
  );
}

export default function BlogListPage(props) {
  return (
    <HtmlClassNameProvider
      className={clsx(
        ThemeClassNames.wrapper.blogPages,
        ThemeClassNames.page.blogListPage
      )}
    >
      <BlogListPageMetadata {...props} />
      <BlogListPageStructuredData {...props} />
      <BlogListPageContent {...props} />
    </HtmlClassNameProvider>
  );
}
