import React from "react";
import clsx from "clsx";
import {
  HtmlClassNameProvider,
  PageMetadata,
  ThemeClassNames,
} from "@docusaurus/theme-common";
import BlogCollectionHero from "@site/src/components/BlogCollectionHero";
import BlogListPaginator from "@theme/BlogListPaginator";
import BlogPostItems from "@theme/BlogPostItems";
import Unlisted from "@theme/ContentVisibility/Unlisted";
import Layout from "@theme/Layout";
import SearchMetadata from "@theme/SearchMetadata";
import { useIsChinese } from "@site/src/components/ds";
import styles from "./styles.module.css";

const COPY = {
  en: {
    metaTitle: (label) => `${label} notes`,
    eyebrow: "Topic collection",
    fallback: (label) => `Research notes connected by ${label}.`,
    notes: (n) => (n === 1 ? "field note" : "field notes"),
    curated: "Curated",
    collection: "topic collection",
    articles: (label) => `${label} articles`,
  },
  zh: {
    metaTitle: (label) => `${label} · 笔记`,
    eyebrow: "主题合集",
    fallback: (label) => `与「${label}」相关的研究笔记。`,
    notes: () => "篇笔记",
    curated: "精选",
    collection: "主题合集",
    articles: (label) => `「${label}」相关文章`,
  },
};

export default function BlogTagsPostsPage({ tag, items, listMetadata }) {
  const copy = useIsChinese() ? COPY.zh : COPY.en;
  return (
    <HtmlClassNameProvider
      className={clsx(
        ThemeClassNames.wrapper.blogPages,
        ThemeClassNames.page.blogTagPostListPage
      )}
    >
      <PageMetadata
        title={copy.metaTitle(tag.label)}
        description={tag.description}
      />
      <SearchMetadata tag="blog_tags_posts" />
      <Layout>
        {tag.unlisted && <Unlisted />}
        <main className={styles.page}>
          <BlogCollectionHero
            compact
            eyebrow={copy.eyebrow}
            title={tag.label}
            description={
              tag.description || copy.fallback(tag.label)
            }
            stats={[
              {
                value: String(tag.count),
                label: copy.notes(tag.count),
              },
              { value: copy.curated, label: copy.collection },
            ]}
          />

          <section
            className={styles.collection}
            aria-label={copy.articles(tag.label)}
          >
            <div className={styles.grid}>
              <BlogPostItems items={items} />
            </div>
            <div className={styles.paginator}>
              <BlogListPaginator metadata={listMetadata} />
            </div>
          </section>
        </main>
      </Layout>
    </HtmlClassNameProvider>
  );
}
