/**
 * Research project template (design/DESIGN_SPEC.md §6.3): layer badges,
 * status and year; title; one-sentence finding; evidence bar; cover figure;
 * then the MDX body (problem, method, results, limitations…) with a TOC.
 */
import React from "react";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import Layout from "@theme/Layout";
import MDXContent from "@theme/MDXContent";
import TOC from "@theme/TOC";
import ContentVisibility from "@theme/ContentVisibility";
import { PageMetadata } from "@docusaurus/theme-common";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { getProject, PROJECT_STATUS } from "@site/src/data/projects";
import { getPublication, doiUrl } from "@site/src/data/publications";
import { APP_MANIFEST } from "@site/src/data/appManifest";
import {
  EvidenceBar,
  JsonLd,
  LayerBadges,
  useIsChinese,
  useLocalize,
} from "@site/src/components/ds";
import styles from "./styles.module.css";

const COPY = {
  en: {
    back: "Research",
    figure: "Figure 1.",
    related: "Related",
    note: "Read the full research note",
    tools: "Tools for this layer",
    onThisPage: "On this page",
  },
  zh: {
    back: "研究",
    figure: "图 1.",
    related: "相关内容",
    note: "阅读完整研究笔记",
    tools: "这一层的相关工具",
    onThisPage: "本页内容",
  },
};

export default function ProjectLayout({ content: MDXPageContent }) {
  const { metadata, assets } = MDXPageContent;
  const { frontMatter } = metadata;
  const isChinese = useIsChinese();
  const localize = useLocalize();
  const copy = isChinese ? COPY.zh : COPY.en;
  const project = getProject(frontMatter.project);
  const siteUrl = useBaseUrl("/", { absolute: true });
  const coverUrl = useBaseUrl(project?.cover ?? "/");

  if (!project) {
    throw new Error(
      `Unknown project id "${frontMatter.project}" in ${metadata.source}`
    );
  }

  const title = localize(project.title);
  const finding = localize(project.finding);
  const status = localize(PROJECT_STATUS[project.status]);
  const cover = assets.image ?? project.cover;
  const relatedTools = APP_MANIFEST.filter(
    (app) =>
      app.layers?.some((layer) => project.layers.includes(layer)) &&
      app.layers.length
  ).slice(0, 4);
  const pubs = project.publications.map(getPublication).filter(Boolean);

  const structuredData = {
    "@context": "https://schema.org",
    "@type": "ResearchProject",
    name: title,
    description: finding,
    url: `${siteUrl}research/${project.id}`,
    founder: { "@type": "Person", name: "Liangchao Deng" },
    ...(pubs.length
      ? {
          subjectOf: pubs.map((pub) => ({
            "@type":
              pub.type === "software"
                ? "SoftwareSourceCode"
                : "ScholarlyArticle",
            name: pub.title,
            datePublished: String(pub.year),
            identifier: `https://doi.org/${pub.doi}`,
            url: doiUrl(pub.doi),
          })),
        }
      : {}),
  };

  return (
    <Layout>
      <PageMetadata title={title} description={finding} image={cover} />
      <JsonLd data={structuredData} />
      <main className={styles.projectPage}>
        <div className={styles.shell}>
          <ContentVisibility metadata={metadata} />
          <header className={styles.projectHeader}>
            <Link className={styles.backLink} to="/research">
              {copy.back}
            </Link>
            <div className={styles.metaRow}>
              <LayerBadges ids={project.layers} />
              <span className={styles.metaText}>
                {status} · {project.year}
              </span>
            </div>
            <Heading as="h1" className={styles.projectTitle}>
              {title}
            </Heading>
            <p className={styles.finding}>{finding}</p>
            <EvidenceBar
              publications={project.publications}
              code={project.code}
              data={project.data}
              tool={project.tool}
            />
          </header>

          {project.cover && (
            <figure className={styles.cover}>
              <img src={coverUrl} alt={localize(project.coverAlt)} />
              <figcaption>
                <strong>{copy.figure}</strong> {localize(project.coverAlt)}
              </figcaption>
            </figure>
          )}

          <div className={styles.bodyGrid}>
            <article className={styles.body}>
              <MDXContent>
                <MDXPageContent />
              </MDXContent>
            </article>
            {MDXPageContent.toc.length > 0 && (
              <aside className={styles.toc} aria-label={copy.onThisPage}>
                <p>{copy.onThisPage}</p>
                <TOC toc={MDXPageContent.toc} />
              </aside>
            )}
          </div>

          <section className={styles.related} aria-label={copy.related}>
            <Heading as="h2" className={styles.relatedTitle}>
              {copy.related}
            </Heading>
            {project.note && (
              <Link className={styles.noteLink} to={project.note}>
                {copy.note}
              </Link>
            )}
            {relatedTools.length > 0 && (
              <div className={styles.relatedRow}>
                <span className={styles.relatedLabel}>{copy.tools}</span>
                {relatedTools.map((app) => (
                  <Link
                    key={app.id}
                    className={styles.relatedLink}
                    to={app.route}
                  >
                    {localize(app.name)}
                  </Link>
                ))}
              </div>
            )}
          </section>
        </div>
      </main>
    </Layout>
  );
}
