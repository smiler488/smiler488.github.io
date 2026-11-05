import React from "react";
import Link from "@docusaurus/Link";
import clsx from "clsx";
import {
  HtmlClassNameProvider,
  PageMetadata,
  ThemeClassNames,
} from "@docusaurus/theme-common";
import BlogCollectionHero from "@site/src/components/BlogCollectionHero";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import SearchMetadata from "@theme/SearchMetadata";
import { useIsChinese } from "@site/src/components/ds";
import styles from "./styles.module.css";

const COPY = {
  en: {
    title: "Research topics",
    metaDescription:
      "Browse research notes by topic, method, and technical discipline.",
    eyebrow: "Topics",
    description:
      "A focused map of the methods and domains that connect the notebook.",
    topics: "curated topics",
    connections: "topic connections",
    all: "All research topics",
    count: (n) => `${n} ${n === 1 ? "note" : "notes"}`,
    fallback: (n) => `Explore ${n} related research notes.`,
  },
  zh: {
    title: "研究主题",
    metaDescription: "按主题、方法与技术领域浏览研究笔记。",
    eyebrow: "主题",
    description: "连接整本笔记的方法与领域地图。",
    topics: "个主题",
    connections: "次主题关联",
    all: "全部研究主题",
    count: (n) => `${n} 篇`,
    fallback: (n) => `浏览 ${n} 篇相关研究笔记。`,
  },
};

export default function BlogTagsListPage({ tags }) {
  const copy = useIsChinese() ? COPY.zh : COPY.en;
  const sortedTags = [...tags].sort(
    (a, b) => b.count - a.count || a.label.localeCompare(b.label)
  );

  return (
    <HtmlClassNameProvider
      className={clsx(
        ThemeClassNames.wrapper.blogPages,
        ThemeClassNames.page.blogTagsListPage
      )}
    >
      <PageMetadata
        title={copy.title}
        description={copy.metaDescription}
      />
      <SearchMetadata tag="blog_tags_list" />
      <Layout>
        <main className={styles.page}>
          <BlogCollectionHero
            compact
            eyebrow={copy.eyebrow}
            title={copy.title}
            description={copy.description}
            stats={[
              { value: String(tags.length), label: copy.topics },
              {
                value: String(
                  tags.reduce((total, tag) => total + tag.count, 0)
                ),
                label: copy.connections,
              },
            ]}
          />

          <section className={styles.tagGrid} aria-label={copy.all}>
            {sortedTags.map((tag) => (
              <Link
                className={styles.tagCard}
                key={tag.permalink}
                to={tag.permalink}
              >
                <div>
                  {copy.count(tag.count)}
                </div>
                <Heading as="h2">{tag.label}</Heading>
                <p>
                  {tag.description || copy.fallback(tag.count)}
                </p>
              </Link>
            ))}
          </section>
        </main>
      </Layout>
    </HtmlClassNameProvider>
  );
}
