/**
 * Datasets (design/DESIGN_SPEC.md §4.3, D-022): the data behind the notes and
 * projects, read from src/data/datasets.js. Each dataset is also published as
 * schema.org Dataset for dataset search engines.
 */
import React from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { DATASETS, LICENSES } from "@site/src/data/datasets";
import { getPublication, doiUrl } from "@site/src/data/publications";
import { cvIdentity } from "@site/src/data/cvData";
import {
  JsonLd,
  LayerBadges,
  PageHero,
  Stats,
  useIsChinese,
  useLocalize,
} from "@site/src/components/ds";
import styles from "./data.module.css";

const COPY = {
  en: {
    pageTitle: "Data",
    pageDescription:
      "Plant phenotyping datasets by Liangchao Deng: 3D cotton point clouds and canopy meshes with organ labels and leaf optics, and leaf BRDF measurements.",
    eyebrow: "Data",
    title: "The data behind the research.",
    lead: "Point clouds, canopy models and leaf optical measurements from the projects and notes, with their formats documented so they can be read without this site.",
    statDatasets: "datasets",
    statDownload: "downloadable here",
    statLayers: "research layers covered",
    download: "Download",
    onRequest: "Available on request",
    onRequestHint:
      "As stated in the published article, the data are available from the corresponding author upon reasonable request.",
    contact: "Contact",
    format: "File format",
    bounds: "Bounding box",
    note: "Research note",
    project: "Project page",
    paper: "Paper",
    code: "Code",
    license: "License",
    reuse:
      "Files downloadable here are licensed CC BY-NC 4.0: free to reuse for non-commercial work with attribution to Liangchao Deng and a link to this page. For commercial use, or to agree on a citable version, get in touch.",
  },
  zh: {
    pageTitle: "数据",
    pageDescription:
      "邓良超的植物表型数据集：带器官标注与叶片光学属性的棉花三维点云与冠层模型，以及叶片 BRDF 测量数据。",
    eyebrow: "数据",
    title: "研究背后的数据。",
    lead: "来自项目与笔记的点云、冠层模型和叶片光学测量。每份数据都写明了文件格式，离开本站也能读取。",
    statDatasets: "个数据集",
    statDownload: "个可在本站下载",
    statLayers: "个研究层",
    download: "下载",
    onRequest: "可申请获取",
    onRequestHint: "按已发表论文的说明，数据可向通讯作者合理申请获取。",
    contact: "联系",
    format: "文件格式",
    bounds: "包围盒",
    note: "研究笔记",
    project: "项目页",
    paper: "论文",
    code: "代码",
    license: "许可协议",
    reuse:
      "本站可下载的文件采用 CC BY-NC 4.0 许可：非商业用途可自由复用，需署名邓良超并链接本页。商业用途或需要确定可引用的版本，请联系作者。",
  },
};

function formatBytes(bytes) {
  return bytes >= 1e6
    ? `${(bytes / 1e6).toFixed(1)} MB`
    : `${Math.round(bytes / 1e3)} KB`;
}

function formatBounds({ min, max, unit }) {
  const axis = (i) => `${min[i].toFixed(3)}–${max[i].toFixed(3)}`;
  return `x ${axis(0)} · y ${axis(1)} · z ${axis(2)} ${unit}`;
}

function DatasetItem({ dataset, copy, localize }) {
  const paper = dataset.publication && getPublication(dataset.publication);
  const links = [
    dataset.project && { to: dataset.project, label: copy.project },
    dataset.note && { to: dataset.note, label: copy.note },
    paper && { href: doiUrl(paper.doi), label: copy.paper },
    dataset.code && { href: dataset.code, label: copy.code },
  ].filter(Boolean);

  return (
    <li className={styles.item} id={dataset.id}>
      <div className={styles.itemMeta}>
        <span>{dataset.year}</span>
        <LayerBadges ids={dataset.layers} />
      </div>
      <Heading as="h2" className={styles.itemTitle}>
        {localize(dataset.title)}
      </Heading>
      <p className={styles.summary}>{localize(dataset.summary)}</p>
      <p className={styles.records}>
        {localize(dataset.records)}
        {dataset.license && (
          <>
            {" · "}
            {copy.license}{" "}
            <Link href={LICENSES[dataset.license].url}>
              {LICENSES[dataset.license].name}
            </Link>
          </>
        )}
      </p>

      <div className={styles.actions}>
        {dataset.access === "download" ? (
          dataset.files.map((file) => (
            // A file download, not navigation: <Link> would route it
            // eslint-disable-next-line @docusaurus/no-html-links
            <a
              key={file.href}
              className={styles.primary}
              href={file.href}
              download
            >
              {copy.download} {file.href.split("/").pop()} ·{" "}
              {formatBytes(file.bytes)}
            </a>
          ))
        ) : (
          <span className={styles.badge}>{copy.onRequest}</span>
        )}
        {links.map((link) => (
          <Link
            key={link.label}
            className={styles.secondary}
            {...(link.to ? { to: link.to } : { href: link.href })}
          >
            {link.label}
          </Link>
        ))}
      </div>

      {dataset.access === "request" && (
        <p className={styles.hint}>{copy.onRequestHint}</p>
      )}

      {dataset.format && (
        <details className={styles.details}>
          <summary>{copy.format}</summary>
          <p>{localize(dataset.format)}</p>
          {dataset.bounds && (
            <p>
              <strong>{copy.bounds}:</strong> {formatBounds(dataset.bounds)}
            </p>
          )}
        </details>
      )}
    </li>
  );
}

export default function DataPage() {
  const isChinese = useIsChinese();
  const localize = useLocalize();
  const copy = isChinese ? COPY.zh : COPY.en;
  const siteUrl = useBaseUrl("/", { absolute: true });
  const origin = siteUrl.replace(/\/(zh-Hans\/)?$/, "");
  const layers = new Set(DATASETS.flatMap((d) => d.layers));

  return (
    <Layout title={copy.pageTitle} description={copy.pageDescription}>
      {DATASETS.map((dataset) => (
        <JsonLd
          key={dataset.id}
          data={{
            "@context": "https://schema.org",
            "@type": "Dataset",
            name: localize(dataset.title),
            description: localize(dataset.summary),
            url: `${siteUrl}data#${dataset.id}`,
            creator: {
              "@type": "Person",
              name: "Liangchao Deng",
              sameAs: [cvIdentity.orcid],
            },
            datePublished: String(dataset.year),
            keywords: ["plant phenotyping", "AI4Plant", ...dataset.layers],
            isAccessibleForFree: dataset.access === "download",
            ...(dataset.license
              ? { license: LICENSES[dataset.license].url }
              : {}),
            ...(dataset.files.length
              ? {
                  distribution: dataset.files.map((file) => ({
                    "@type": "DataDownload",
                    encodingFormat: "application/octet-stream",
                    contentUrl: `${origin}${file.href}`,
                    contentSize: formatBytes(file.bytes),
                  })),
                }
              : {}),
            ...(dataset.publication
              ? {
                  citation: doiUrl(getPublication(dataset.publication).doi),
                }
              : {}),
          }}
        />
      ))}
      <main className={styles.page}>
        <div className={styles.shell}>
          <PageHero eyebrow={copy.eyebrow} title={copy.title} lead={copy.lead}>
            <Stats
              items={[
                { value: DATASETS.length, label: copy.statDatasets },
                {
                  value: DATASETS.filter((d) => d.access === "download")
                    .length,
                  label: copy.statDownload,
                },
                { value: layers.size, label: copy.statLayers },
              ]}
            />
          </PageHero>

          <ol className={styles.list}>
            {DATASETS.map((dataset) => (
              <DatasetItem
                key={dataset.id}
                dataset={dataset}
                copy={copy}
                localize={localize}
              />
            ))}
          </ol>

          <p className={styles.reuse}>
            {copy.reuse} <Link to="/cv#contact-channels">{copy.contact}</Link>
          </p>
        </div>
      </main>
    </Layout>
  );
}
