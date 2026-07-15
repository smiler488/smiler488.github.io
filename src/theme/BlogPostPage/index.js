import React from "react";
import clsx from "clsx";
import {
  HtmlClassNameProvider,
  ThemeClassNames,
} from "@docusaurus/theme-common";
import {
  BlogPostProvider,
  useBlogPost,
} from "@docusaurus/plugin-content-blog/client";
import Layout from "@theme/Layout";
import BlogPostItem from "@theme/BlogPostItem";
import BlogPostPaginator from "@theme/BlogPostPaginator";
import BlogPostPageMetadata from "@theme/BlogPostPage/Metadata";
import BlogPostPageStructuredData from "@theme/BlogPostPage/StructuredData";
import TOC from "@theme/TOC";
import TOCInline from "@theme/TOCInline";
import Comment from "@site/src/components/comment";
import ContentVisibility from "@theme/ContentVisibility";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

function ArticleToc({ toc, minHeadingLevel, maxHeadingLevel, inline = false }) {
  const TocComponent = inline ? TOCInline : TOC;
  return (
    <TocComponent
      toc={toc}
      minHeadingLevel={minHeadingLevel}
      maxHeadingLevel={maxHeadingLevel}
    />
  );
}

function BlogPostPageContent({ children }) {
  const { metadata, toc } = useBlogPost();
  const { nextItem, prevItem, frontMatter } = metadata;
  const {
    hide_table_of_contents: hideTableOfContents,
    toc_min_heading_level: tocMinHeadingLevel,
    toc_max_heading_level: tocMaxHeadingLevel,
  } = frontMatter;
  const showToc = !hideTableOfContents && toc.length > 0;

  return (
    <Layout>
      <ContentVisibility metadata={metadata} />
      <main className={styles.page}>
        <div
          className={clsx(styles.layout, !showToc && styles.layoutWithoutToc)}
        >
          <div className={styles.articleColumn}>
            {showToc && (
              <details className={styles.mobileToc}>
                <summary>
                  <span>On this page</span>
                  <span aria-hidden="true">＋</span>
                </summary>
                <div className={styles.mobileTocContent}>
                  <ArticleToc
                    toc={toc}
                    minHeadingLevel={tocMinHeadingLevel}
                    maxHeadingLevel={tocMaxHeadingLevel}
                    inline
                  />
                </div>
              </details>
            )}

            <BlogPostItem>{children}</BlogPostItem>

            {(nextItem || prevItem) && (
              <div className={styles.paginatorSurface}>
                <BlogPostPaginator nextItem={nextItem} prevItem={prevItem} />
              </div>
            )}

            <section className={styles.commentSurface} aria-label="Discussion">
              <div>
                <p>DISCUSSION</p>
                <Heading as="h2">Questions or field notes?</Heading>
              </div>
              <Comment />
            </section>
          </div>

          {showToc && (
            <aside className={styles.tocColumn} aria-label="Table of contents">
              <div className={styles.tocSurface}>
                <p>ON THIS PAGE</p>
                <ArticleToc
                  toc={toc}
                  minHeadingLevel={tocMinHeadingLevel}
                  maxHeadingLevel={tocMaxHeadingLevel}
                />
              </div>
            </aside>
          )}
        </div>
      </main>
    </Layout>
  );
}

export default function BlogPostPage(props) {
  const BlogPostContent = props.content;
  return (
    <BlogPostProvider content={props.content} isBlogPostPage>
      <HtmlClassNameProvider
        className={clsx(
          ThemeClassNames.wrapper.blogPages,
          ThemeClassNames.page.blogPostPage
        )}
      >
        <BlogPostPageMetadata />
        <BlogPostPageStructuredData />
        <BlogPostPageContent>
          <BlogPostContent />
        </BlogPostPageContent>
      </HtmlClassNameProvider>
    </BlogPostProvider>
  );
}
