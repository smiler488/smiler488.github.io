import React from "react";
import Link from "@docusaurus/Link";
import { PageMetadata } from "@docusaurus/theme-common";
import BlogCollectionHero from "@site/src/components/BlogCollectionHero";
import Heading from "@theme/Heading";
import Layout from "@theme/Layout";
import { useIsChinese } from "@site/src/components/ds";
import styles from "./styles.module.css";

const MONTH_FORMAT = { day: "numeric", month: "short", timeZone: "UTC" };
const FORMATTERS = {
  en: new Intl.DateTimeFormat("en", MONTH_FORMAT),
  zh: new Intl.DateTimeFormat("zh-CN", MONTH_FORMAT),
};

const COPY = {
  en: {
    title: "Research archive",
    metaDescription:
      "A chronological archive of research notes, technical guides, and project records.",
    eyebrow: "Archive",
    description:
      "Every field note in one calm timeline—from developer workflows to AI-assisted plant phenotyping.",
    notes: "field notes",
    years: "years documented",
    byYear: "Blog archive by year",
    yearCount: (n) => `${n} ${n === 1 ? "note" : "notes"}`,
  },
  zh: {
    title: "研究归档",
    metaDescription: "按时间排列的研究笔记、技术指南与项目记录归档。",
    eyebrow: "归档",
    description: "所有笔记按时间排成一条线，从开发工作流到 AI 辅助的植物表型。",
    notes: "篇笔记",
    years: "年的记录",
    byYear: "按年份浏览笔记归档",
    yearCount: (n) => `${n} 篇`,
  },
};

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
  const isChinese = useIsChinese();
  const copy = isChinese ? COPY.zh : COPY.en;
  const monthFormatter = FORMATTERS[isChinese ? "zh" : "en"];
  const years = Array.from(groupPostsByYear(archive.blogPosts).entries()).sort(
    ([yearA], [yearB]) => Number(yearB) - Number(yearA)
  );

  return (
    <>
      <PageMetadata
        title={copy.title}
        description={copy.metaDescription}
      />
      <Layout>
        <main className={styles.page}>
          <BlogCollectionHero
            compact
            eyebrow={copy.eyebrow}
            title={copy.title}
            description={copy.description}
            stats={[
              { value: String(archive.blogPosts.length), label: copy.notes },
              { value: String(years.length), label: copy.years },
            ]}
          />

          <section
            className={styles.timeline}
            aria-label={copy.byYear}
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
                  <span>{copy.yearCount(posts.length)}</span>
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
