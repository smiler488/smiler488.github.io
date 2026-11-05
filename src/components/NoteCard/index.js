/**
 * One note in the /blog grid. Presentational, so the paginated list (blog
 * post context) and the filtered view (site-index global data) share a card.
 */
import React from "react";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import { useIsChinese } from "@site/src/components/ds";
import styles from "@site/src/theme/BlogPostItem/styles.module.css";

const FORMATTERS = {
  en: new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }),
  zh: new Intl.DateTimeFormat("zh-CN", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }),
};

export default function NoteCard({
  title,
  description,
  permalink,
  date,
  image,
  category,
  tags = [],
}) {
  const isChinese = useIsChinese();
  const label = category || (isChinese ? "研究笔记" : "Research note");
  const useImage = image && !image.includes("blog-default");
  const formatted = FORMATTERS[isChinese ? "zh" : "en"].format(new Date(date));

  return (
    <article className={styles.card}>
      <Link
        className={styles.visualLink}
        to={permalink}
        aria-label={isChinese ? `阅读：${title}` : `Read ${title}`}
      >
        <div className={styles.visual} aria-hidden="true">
          {useImage ? (
            <img src={image} alt="" loading="lazy" />
          ) : (
            <div className={styles.typeCover}>
              <span>{label}</span>
            </div>
          )}
        </div>
      </Link>
      <div className={styles.cardBody}>
        <div className={styles.cardTypeRow}>
          <span>{label}</span>
          <span aria-hidden="true">·</span>
          <time dateTime={date}>{formatted}</time>
        </div>
        <Heading as="h2">
          <Link to={permalink}>{title}</Link>
        </Heading>
        <p className={styles.cardDescription}>{description}</p>
        {tags.length > 0 && (
          <div
            className={styles.cardTags}
            aria-label={isChinese ? "主题" : "Topics"}
          >
            {tags.slice(0, 3).map((tag) => (
              <Link key={tag.permalink} to={tag.permalink}>
                {tag.label}
              </Link>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}
