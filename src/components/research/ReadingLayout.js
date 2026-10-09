/**
 * Unboxed reading layout for MDX pages with `layout: reading`
 * (Now, Open problems). Front matter: title, description, eyebrow, updated.
 */
import React from "react";
import Heading from "@theme/Heading";
import Layout from "@theme/Layout";
import MDXContent from "@theme/MDXContent";
import ContentVisibility from "@theme/ContentVisibility";
import { PageMetadata } from "@docusaurus/theme-common";
import styles from "./styles.module.css";

export default function ReadingLayout({ content: MDXPageContent }) {
  const { metadata } = MDXPageContent;
  const { title, description, frontMatter } = metadata;

  return (
    <Layout>
      <PageMetadata title={title} description={description} />
      <main className={styles.readingPage}>
        <div className={styles.readingShell}>
          <ContentVisibility metadata={metadata} />
          <header className={styles.readingHeader}>
            {frontMatter.eyebrow && (
              <p className={styles.eyebrow}>{frontMatter.eyebrow}</p>
            )}
            <Heading as="h1" className={styles.projectTitle}>
              {title}
            </Heading>
            {description && <p className={styles.finding}>{description}</p>}
            {frontMatter.updated && (
              <p className={styles.metaText}>{frontMatter.updated}</p>
            )}
          </header>
          <article className={styles.body}>
            <MDXContent>
              <MDXPageContent />
            </MDXContent>
          </article>
        </div>
      </main>
    </Layout>
  );
}
