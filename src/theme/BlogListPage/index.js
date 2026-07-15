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

function BlogListPageContent({ metadata, items }) {
  const articleLabel = metadata.totalCount === 1 ? "field note" : "field notes";
  const pageLabel =
    metadata.totalPages > 1 ? `${metadata.totalPages} pages` : "one collection";

  return (
    <Layout>
      <main className={styles.page}>
        <BlogCollectionHero
          stats={[
            { value: String(metadata.totalCount), label: articleLabel },
            {
              value: `2021–${new Date().getUTCFullYear()}`,
              label: "research timeline",
            },
            { value: pageLabel, label: "curated reading" },
          ]}
        />

        <section className={styles.collection} aria-labelledby="latest-notes">
          <div className={styles.collectionHeader}>
            <div>
              <p className={styles.kicker}>LATEST NOTES</p>
              <Heading as="h2" id="latest-notes">
                Ideas, methods, and working systems
              </Heading>
            </div>
            <p>
              Concise project records and technical guides, revised to separate
              verified results from experimental workflows.
            </p>
          </div>

          <div className={styles.grid}>
            <BlogPostItems items={items} />
          </div>
          <div className={styles.paginator}>
            <BlogListPaginator metadata={metadata} />
          </div>
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
