/**
 * MDXPage with two site layouts selected by front matter
 * (design/DESIGN_SPEC.md §6):
 *   project: <id>    → research project template (§6.3)
 *   layout: reading  → unboxed reading column (Now, Open problems)
 * Any other MDX page falls back to the stock Docusaurus layout.
 */
import React from "react";
import clsx from "clsx";
import {
  PageMetadata,
  HtmlClassNameProvider,
  ThemeClassNames,
} from "@docusaurus/theme-common";
import Layout from "@theme/Layout";
import MDXContent from "@theme/MDXContent";
import TOC from "@theme/TOC";
import ContentVisibility from "@theme/ContentVisibility";
import EditMetaRow from "@theme/EditMetaRow";
import ProjectLayout from "@site/src/components/research/ProjectLayout";
import ReadingLayout from "@site/src/components/research/ReadingLayout";
import styles from "./styles.module.css";

function DefaultMDXPage({ content: MDXPageContent }) {
  const { metadata, assets } = MDXPageContent;
  const {
    title,
    editUrl,
    description,
    frontMatter,
    lastUpdatedBy,
    lastUpdatedAt,
  } = metadata;
  const { keywords, hide_table_of_contents: hideTableOfContents } = frontMatter;
  const image = assets.image ?? frontMatter.image;
  const canDisplayEditMetaRow = !!(editUrl || lastUpdatedAt || lastUpdatedBy);
  return (
    <Layout>
      <PageMetadata
        title={title}
        description={description}
        keywords={keywords}
        image={image}
      />
      <main className="container container--fluid margin-vert--lg">
        <div className={clsx("row", styles.mdxPageWrapper)}>
          <div className={clsx("col", !hideTableOfContents && "col--8")}>
            <ContentVisibility metadata={metadata} />
            <article>
              <MDXContent>
                <MDXPageContent />
              </MDXContent>
            </article>
            {canDisplayEditMetaRow && (
              <EditMetaRow
                className={clsx(
                  "margin-top--sm",
                  ThemeClassNames.pages.pageFooterEditMetaRow
                )}
                editUrl={editUrl}
                lastUpdatedAt={lastUpdatedAt}
                lastUpdatedBy={lastUpdatedBy}
              />
            )}
          </div>
          {!hideTableOfContents && MDXPageContent.toc.length > 0 && (
            <div className="col col--2">
              <TOC
                toc={MDXPageContent.toc}
                minHeadingLevel={frontMatter.toc_min_heading_level}
                maxHeadingLevel={frontMatter.toc_max_heading_level}
              />
            </div>
          )}
        </div>
      </main>
    </Layout>
  );
}

export default function MDXPage(props) {
  const { content: MDXPageContent } = props;
  const { frontMatter } = MDXPageContent.metadata;
  const { wrapperClassName } = frontMatter;

  let page;
  if (frontMatter.project) {
    page = <ProjectLayout content={MDXPageContent} />;
  } else if (frontMatter.layout === "reading") {
    page = <ReadingLayout content={MDXPageContent} />;
  } else {
    page = <DefaultMDXPage content={MDXPageContent} />;
  }

  return (
    <HtmlClassNameProvider
      className={clsx(
        wrapperClassName ?? ThemeClassNames.wrapper.mdxPages,
        ThemeClassNames.page.mdxPage
      )}
    >
      {page}
    </HtmlClassNameProvider>
  );
}
