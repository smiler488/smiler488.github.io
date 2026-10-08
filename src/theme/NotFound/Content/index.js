import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import styles from "./styles.module.css";

const COPY = {
  en: {
    code: "404",
    title: "This page isn't here.",
    text: "The link may be outdated, or the page has moved. These are the places most people are looking for.",
    links: [
      { to: "/", label: "Home" },
      { to: "/blog", label: "Research notes" },
      { to: "/app", label: "App Lab" },
      { to: "/cv", label: "Curriculum vitae" },
    ],
  },
  zh: {
    code: "404",
    title: "这个页面不存在。",
    text: "链接可能已过期，或页面已移动。以下是大多数访客要找的地方。",
    links: [
      { to: "/", label: "首页" },
      { to: "/blog", label: "研究笔记" },
      { to: "/app", label: "App Lab" },
      { to: "/cv", label: "个人简历" },
    ],
  },
};

export default function NotFoundContent({ className }) {
  const { i18n } = useDocusaurusContext();
  const copy = i18n.currentLocale === "zh-Hans" ? COPY.zh : COPY.en;

  return (
    <main className={clsx(styles.page, className)}>
      <p className={styles.code}>{copy.code}</p>
      <Heading as="h1" className={styles.title}>
        {copy.title}
      </Heading>
      <p className={styles.text}>{copy.text}</p>
      <nav className={styles.links} aria-label={copy.title}>
        {copy.links.map((link, index) => (
          <Link
            key={link.to}
            to={link.to}
            className={index === 0 ? styles.primary : styles.secondary}
          >
            {link.label}
          </Link>
        ))}
      </nav>
    </main>
  );
}
