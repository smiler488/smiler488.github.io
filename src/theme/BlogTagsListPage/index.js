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
import styles from "./styles.module.css";

export default function BlogTagsListPage({ tags }) {
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
        title="Research topics"
        description="Browse research notes by topic, method, and technical discipline."
      />
      <SearchMetadata tag="blog_tags_list" />
      <Layout>
        <main className={styles.page}>
          <BlogCollectionHero
            compact
            eyebrow="TOPIC INDEX"
            title="Research topics"
            description="A focused map of the methods and domains that connect the notebook."
            stats={[
              { value: String(tags.length), label: "curated topics" },
              {
                value: String(
                  tags.reduce((total, tag) => total + tag.count, 0)
                ),
                label: "topic connections",
              },
            ]}
          />

          <section className={styles.tagGrid} aria-label="All research topics">
            {sortedTags.map((tag) => (
              <Link
                className={styles.tagCard}
                key={tag.permalink}
                to={tag.permalink}
              >
                <div>
                  <span>{String(tag.count).padStart(2, "0")}</span>
                  <span aria-hidden="true">↗</span>
                </div>
                <Heading as="h2">{tag.label}</Heading>
                <p>
                  {tag.description ||
                    `Explore ${tag.count} related research notes.`}
                </p>
              </Link>
            ))}
          </section>
        </main>
      </Layout>
    </HtmlClassNameProvider>
  );
}
