import React from "react";
import Link from "@docusaurus/Link";
import { PageMetadata } from "@docusaurus/theme-common";
import BlogCollectionHero from "@site/src/components/BlogCollectionHero";
import Heading from "@theme/Heading";
import Layout from "@theme/Layout";
import styles from "./styles.module.css";

const monthFormatter = new Intl.DateTimeFormat("en", {
  day: "numeric",
  month: "short",
  timeZone: "UTC",
});

function groupPostsByYear(blogPosts) {
  return blogPosts.reduce((years, post) => {
    const year = post.metadata.date.slice(0, 4);
    const current = years.get(year) || [];
    current.push(post);
    current.sort(
      (a, b) => new Date(b.metadata.date) - new Date(a.metadata.date)
    );
    years.set(year, current);
    return years;
  }, new Map());
}

export default function BlogArchivePage({ archive }) {
  const years = Array.from(groupPostsByYear(archive.blogPosts).entries()).sort(
    ([yearA], [yearB]) => Number(yearB) - Number(yearA)
  );

  return (
    <>
      <PageMetadata
        title="Research archive"
        description="A chronological archive of research notes, technical guides, and project records."
      />
      <Layout>
        <main className={styles.page}>
          <BlogCollectionHero
            compact
            eyebrow="CHRONOLOGICAL INDEX"
            title="Research archive"
            description="Every field note in one calm timeline—from developer workflows to AI-assisted plant phenotyping."
            stats={[
              { value: String(archive.blogPosts.length), label: "field notes" },
              { value: String(years.length), label: "years documented" },
            ]}
          />

          <section
            className={styles.timeline}
            aria-label="Blog archive by year"
          >
            {years.map(([year, posts]) => (
              <section
                className={styles.year}
                key={year}
                aria-labelledby={`year-${year}`}
              >
                <div className={styles.yearHeading}>
                  <Heading as="h2" id={`year-${year}`}>
                    {year}
                  </Heading>
                  <span>{posts.length} notes</span>
                </div>
                <ol>
                  {posts.map((post) => {
                    const { metadata } = post;
                    return (
                      <li key={metadata.permalink}>
                        <time dateTime={metadata.date}>
                          {monthFormatter.format(new Date(metadata.date))}
                        </time>
                        <div>
                          <Link to={metadata.permalink}>{metadata.title}</Link>
                          <p>{metadata.description}</p>
                        </div>
                        <span aria-hidden="true">↗</span>
                      </li>
                    );
                  })}
                </ol>
              </section>
            ))}
          </section>
        </main>
      </Layout>
    </>
  );
}
