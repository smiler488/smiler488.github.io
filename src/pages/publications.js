/**
 * Publications and citable software (design/DESIGN_SPEC.md §6.4), read from
 * the single source in src/data/publications.js.
 */
import React from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import useBaseUrl from "@docusaurus/useBaseUrl";
import {
  PUBLICATIONS,
  PUBLICATION_KIND,
  doiUrl,
} from "@site/src/data/publications";
import { getProject } from "@site/src/data/projects";
import {
  EvidenceBar,
  JsonLd,
  LayerBadges,
  PageHero,
  Stats,
  useIsChinese,
  useLocalize,
} from "@site/src/components/ds";
import styles from "./publications.module.css";

const SELF = "Deng, L.";

const COPY = {
  en: {
    pageTitle: "Publications",
    pageDescription:
      "Peer-reviewed papers and citable research software by Liangchao Deng, with DOIs, code and citation formats.",
    eyebrow: "Publications",
    title: "Papers and citable software.",
    lead: "Peer-reviewed research and archived software, each with a DOI, links to code where available, and ready-to-copy citations.",
    statTotal: "outputs",
    statArticles: "peer-reviewed articles",
    statSoftware: "software releases",
    project: "Project page",
    profiles: "Full record",
  },
  zh: {
    pageTitle: "论文",
    pageDescription:
      "邓良超的同行评议论文与可引用科研软件，附 DOI、代码与引用格式。",
    eyebrow: "论文与软件",
    title: "论文与可引用的软件。",
    lead: "同行评议研究与已存档的科研软件。每一项都有 DOI，附可用的代码链接和可直接复制的引用格式。",
    statTotal: "项成果",
    statArticles: "篇同行评议论文",
    statSoftware: "个软件版本",
    project: "项目页",
    profiles: "完整记录",
  },
};

function Authors({ authors }) {
  return (
    <p className={styles.authors}>
      {authors.map((name, index) => (
        <React.Fragment key={name}>
          {index > 0 && ", "}
          {name === SELF ? <strong>{name}</strong> : name}
        </React.Fragment>
      ))}
    </p>
  );
}

export default function PublicationsPage() {
  const isChinese = useIsChinese();
  const localize = useLocalize();
  const copy = isChinese ? COPY.zh : COPY.en;
  const siteUrl = useBaseUrl("/", { absolute: true });
  const locale = isChinese ? "zh" : "en";

  const years = [...new Set(PUBLICATIONS.map((p) => p.year))].sort(
    (a, b) => b - a
  );

  return (
    <Layout title={copy.pageTitle} description={copy.pageDescription}>
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "ItemList",
          name: copy.pageTitle,
          url: `${siteUrl}publications`,
          itemListElement: PUBLICATIONS.map((pub, index) => ({
            "@type": "ListItem",
            position: index + 1,
            item: {
              "@type":
                pub.type === "software"
                  ? "SoftwareSourceCode"
                  : "ScholarlyArticle",
              name: pub.title,
              headline: pub.title,
              datePublished: String(pub.year),
              author: pub.authors.map((name) => ({ "@type": "Person", name })),
              ...(pub.type === "article"
                ? { isPartOf: { "@type": "Periodical", name: pub.venue } }
                : { publisher: { "@type": "Organization", name: pub.venue } }),
              identifier: {
                "@type": "PropertyValue",
                propertyID: "DOI",
                value: pub.doi,
              },
              url: doiUrl(pub.doi),
            },
          })),
        }}
      />
      <main className={styles.page}>
        <div className={styles.shell}>
          <PageHero eyebrow={copy.eyebrow} title={copy.title} lead={copy.lead}>
            <Stats
              items={[
                { value: PUBLICATIONS.length, label: copy.statTotal },
                {
                  value: PUBLICATIONS.filter((p) => p.type === "article")
                    .length,
                  label: copy.statArticles,
                },
                {
                  value: PUBLICATIONS.filter((p) => p.type === "software")
                    .length,
                  label: copy.statSoftware,
                },
              ]}
            />
          </PageHero>

          {years.map((year) => (
            <section
              key={year}
              className={styles.year}
              aria-labelledby={`year-${year}`}
            >
              <Heading as="h2" id={`year-${year}`} className={styles.yearTitle}>
                {year}
              </Heading>
              <ol className={styles.list}>
                {PUBLICATIONS.filter((p) => p.year === year).map((pub) => {
                  const project = pub.project && getProject(pub.project);
                  return (
                    <li key={pub.id} className={styles.item}>
                      <div className={styles.itemMeta}>
                        <span>{PUBLICATION_KIND[pub.type][locale]}</span>
                        <span aria-hidden="true">·</span>
                        <span>{pub.venue}</span>
                        <LayerBadges ids={pub.layers} />
                      </div>
                      <Heading as="h3" className={styles.itemTitle}>
                        <Link href={doiUrl(pub.doi)}>{pub.title}</Link>
                      </Heading>
                      <Authors authors={pub.authors} />
                      {pub.description && (
                        <p className={styles.description}>
                          {localize(pub.description)}
                        </p>
                      )}
                      <div className={styles.actions}>
                        {project && (
                          <Link
                            className={styles.projectLink}
                            to={`/research/${project.id}`}
                          >
                            {copy.project}
                          </Link>
                        )}
                        <EvidenceBar publications={[pub.id]} code={pub.code} />
                      </div>
                    </li>
                  );
                })}
              </ol>
            </section>
          ))}
        </div>
      </main>
    </Layout>
  );
}
