import React from "react";
import Link from "@docusaurus/Link";
import { useBlogPost } from "@docusaurus/plugin-content-blog/client";
import BlogPostItemContent from "@theme/BlogPostItem/Content";
import BlogPostItemFooter from "@theme/BlogPostItem/Footer";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

const dateFormatter = new Intl.DateTimeFormat("en", {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});

function formatDate(date) {
  return dateFormatter.format(new Date(date));
}

function ReadingMeta({ date, readingTime }) {
  return (
    <div className={styles.readingMeta}>
      <time dateTime={date}>{formatDate(date)}</time>
      {typeof readingTime === "number" && (
        <>
          <span aria-hidden="true">·</span>
          <span>{Math.max(1, Math.ceil(readingTime))} min read</span>
        </>
      )}
    </div>
  );
}

function TopicVisual({ image, category }) {
  const useImage = image && !image.includes("blog-default");
  const initials = (category || "Research note")
    .split(/\s|&/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0])
    .join("");

  return (
    <div className={styles.visual} aria-hidden="true">
      {useImage ? (
        <img src={image} alt="" loading="lazy" />
      ) : (
        <div className={styles.abstractVisual}>
          <span className={styles.orbitOne} />
          <span className={styles.orbitTwo} />
          <strong>{initials}</strong>
        </div>
      )}
      <span className={styles.visualLabel}>{category || "Research note"}</span>
    </div>
  );
}

function ListCard() {
  const { metadata, assets } = useBlogPost();
  const {
    title,
    description,
    permalink,
    date,
    readingTime,
    tags,
    frontMatter,
  } = metadata;
  const category = frontMatter.category || "Research note";
  const articleType = frontMatter.article_type || "Field note";

  return (
    <article className={styles.card}>
      <Link
        className={styles.visualLink}
        to={permalink}
        aria-label={`Read ${title}`}
      >
        <TopicVisual
          image={assets.image || frontMatter.image}
          category={category}
        />
      </Link>
      <div className={styles.cardBody}>
        <div className={styles.cardTypeRow}>
          <span>{articleType}</span>
          <ReadingMeta date={date} readingTime={readingTime} />
        </div>
        <Heading as="h2">
          <Link to={permalink}>{title}</Link>
        </Heading>
        <p className={styles.cardDescription}>{description}</p>
        <footer className={styles.cardFooter}>
          <div className={styles.cardTags} aria-label="Topics">
            {tags.slice(0, 3).map((tag) => (
              <Link key={tag.permalink} to={tag.permalink}>
                {tag.label}
              </Link>
            ))}
          </div>
          <Link
            className={styles.readLink}
            to={permalink}
            aria-label={`Read ${title}`}
          >
            <span>Read</span>
            <span aria-hidden="true">↗</span>
          </Link>
        </footer>
      </div>
    </article>
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
  const category = frontMatter.category || "Research note";
  const articleType = frontMatter.article_type || "Field note";
  const author = authors[0];
  const authorImage = assets.authorsImageUrls?.[0] || author?.imageURL;

  return (
    <article className={styles.postArticle}>
      <header className={styles.postHero}>
        <div className={styles.postGlow} aria-hidden="true" />
        <Link className={styles.backLink} to="/blog">
          <span aria-hidden="true">←</span> Research notebook
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
          <div className={styles.heroTags} aria-label="Article topics">
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
