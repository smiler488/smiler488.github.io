import React from "react";
import Link from "@docusaurus/Link";
import { useBlogPost } from "@docusaurus/plugin-content-blog/client";
import BlogPostItemContent from "@theme/BlogPostItem/Content";
import BlogPostItemFooter from "@theme/BlogPostItem/Footer";
import Heading from "@theme/Heading";
import NoteCard from "@site/src/components/NoteCard";
import { useIsChinese } from "@site/src/components/ds";
import styles from "./styles.module.css";

const DATE_FORMAT = {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
};

const COPY = {
  en: {
    formatter: new Intl.DateTimeFormat("en", DATE_FORMAT),
    minutes: (n) => `${n} min read`,
    notebook: "Research notebook",
    category: "Research note",
    articleType: "Field note",
    topics: "Article topics",
  },
  zh: {
    formatter: new Intl.DateTimeFormat("zh-CN", DATE_FORMAT),
    minutes: (n) => `阅读约 ${n} 分钟`,
    notebook: "研究笔记本",
    category: "研究笔记",
    articleType: "田间笔记",
    topics: "文章主题",
  },
};

function useCopy() {
  return useIsChinese() ? COPY.zh : COPY.en;
}

function ReadingMeta({ date, readingTime }) {
  const copy = useCopy();
  return (
    <div className={styles.readingMeta}>
      <time dateTime={date}>{copy.formatter.format(new Date(date))}</time>
      {typeof readingTime === "number" && (
        <>
          <span aria-hidden="true">·</span>
          <span>{copy.minutes(Math.max(1, Math.ceil(readingTime)))}</span>
        </>
      )}
    </div>
  );
}

function ListCard() {
  const { metadata, assets } = useBlogPost();
  const { title, description, permalink, date, tags, frontMatter } = metadata;
  return (
    <NoteCard
      title={title}
      description={description}
      permalink={permalink}
      date={date}
      image={assets.image || frontMatter.image}
      category={frontMatter.category}
      tags={tags}
    />
  );
}

function Author({ author, imageUrl }) {
  if (!author) {
    return null;
  }

  const identity = (
    <>
      {imageUrl && <img src={imageUrl} alt="" />}
      <span>
        <strong>{author.name}</strong>
        {author.title && <small>{author.title}</small>}
      </span>
    </>
  );

  return author.url ? (
    <Link className={styles.author} to={author.url}>
      {identity}
    </Link>
  ) : (
    <div className={styles.author}>{identity}</div>
  );
}

function PostArticle({ children }) {
  const { metadata, assets } = useBlogPost();
  const { title, description, date, readingTime, tags, authors, frontMatter } =
    metadata;
  const copy = useCopy();
  const category = frontMatter.category || copy.category;
  const articleType = frontMatter.article_type || copy.articleType;
  const author = authors[0];
  const authorImage = assets.authorsImageUrls?.[0] || author?.imageURL;

  return (
    <article className={styles.postArticle}>
      <header className={styles.postHero}>
        <Link className={styles.backLink} to="/blog">
          {copy.notebook}
        </Link>
        <div className={styles.postLabels}>
          <span>{category}</span>
          <span>{articleType}</span>
        </div>
        <Heading as="h1">{title}</Heading>
        <p className={styles.postDescription}>{description}</p>
        <div className={styles.heroMeta}>
          <Author author={author} imageUrl={authorImage} />
          <ReadingMeta date={date} readingTime={readingTime} />
        </div>
        {tags.length > 0 && (
          <div className={styles.heroTags} aria-label={copy.topics}>
            {tags.map((tag) => (
              <Link key={tag.permalink} to={tag.permalink}>
                {tag.label}
              </Link>
            ))}
          </div>
        )}
      </header>

      <div className={styles.contentSurface}>
        <BlogPostItemContent className={styles.content}>
          {children}
        </BlogPostItemContent>
        <BlogPostItemFooter />
      </div>
    </article>
  );
}

export default function BlogPostItem({ children }) {
  const { isBlogPostPage } = useBlogPost();
  return isBlogPostPage ? <PostArticle>{children}</PostArticle> : <ListCard />;
}
