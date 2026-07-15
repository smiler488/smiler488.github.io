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
import styles from "./styles.module.css";

export default function BlogTagsPostsPage({ tag, items, listMetadata }) {
  return (
    <HtmlClassNameProvider
      className={clsx(
        ThemeClassNames.wrapper.blogPages,
        ThemeClassNames.page.blogTagPostListPage
      )}
    >
      <PageMetadata
        title={`${tag.label} notes`}
        description={tag.description}
      />
      <SearchMetadata tag="blog_tags_posts" />
      <Layout>
        {tag.unlisted && <Unlisted />}
        <main className={styles.page}>
          <BlogCollectionHero
            compact
            eyebrow="TOPIC COLLECTION"
            title={tag.label}
            description={
              tag.description || `Research notes connected by ${tag.label}.`
            }
            stats={[
              {
                value: String(tag.count),
                label: tag.count === 1 ? "field note" : "field notes",
              },
              { value: "Curated", label: "topic collection" },
            ]}
          />

          <section
            className={styles.collection}
            aria-label={`${tag.label} articles`}
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
