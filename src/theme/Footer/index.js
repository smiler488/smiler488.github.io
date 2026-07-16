import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

const NAV_LINKS = [
  { to: "/", en: "Home", zh: "首页" },
  { to: "/blog", en: "Research notes", zh: "研究笔记" },
  { to: "/app", en: "App Lab", zh: "小程序实验室" },
  { to: "/resources", en: "Learning resources", zh: "学习资源" },
  {
    to: "/docs/category/tutorial---apps",
    en: "App tutorials",
    zh: "小程序教程",
  },
  { to: "/cv", en: "Curriculum vitae", zh: "个人简历" },
];

const CONTACT_LINKS = [
  {
    type: "academic",
    labelEn: "googalphdlc@gmail.com",
    labelZh: "googalphdlc@gmail.com",
    href: "mailto:googalphdlc@gmail.com",
  },
  {
    type: "collaboration",
    labelEn: "dengliangchao@smiler488.com",
    labelZh: "dengliangchao@smiler488.com",
    href: "mailto:dengliangchao@smiler488.com",
  },
  {
    type: "assistant",
    labelEn: "WeChat Support Bot ↗",
    labelZh: "微信咨询机器人 ↗",
    href: "https://work.weixin.qq.com/kfid/kfc63941027aeefc636",
    isExternal: true,
  },
];

const SOCIAL_LINKS = [
  {
    name: "Bilibili",
    icon: "/img/Bilibili.png",
    href: "https://space.bilibili.com/16062789",
    color: "#00a1d6",
  },
  {
    name: "Douyin",
    icon: "/img/Douyin.png",
    href: "https://v.douyin.com/-1moIAdEYpg/",
    color: "#fe2c55",
  },
  {
    name: "Weibo",
    icon: "/img/Weibo.png",
    href: "https://m.weibo.cn/profile/5283742028",
    color: "#e6162d",
  },
  {
    name: "WeChat",
    icon: "/img/WeChat Offical.png",
    href: "https://mp.weixin.qq.com/s/JPLLGnM6fwT8XpBdfoXKNA",
    color: "#07c160",
  },
  {
    name: "YouTube",
    icon: "/img/YouTube.png",
    href: "https://www.youtube.com/channel/UCmz7DQ3nEPRxj4rvEQUCvAg",
    color: "#ff0000",
  },
  {
    name: "TikTok",
    icon: "/img/TikTok.png",
    href: "https://www.tiktok.com/@smiler488tt",
    color: "#25f4ee",
  },
  {
    name: "X",
    icon: "/img/X.png",
    href: "https://x.com/smiler488",
    color: "#6b7280",
  },
  {
    name: "Reddit",
    icon: "/img/Reddit.png",
    href: "https://www.reddit.com/user/smiler488/",
    color: "#ff4500",
  },
  {
    name: "LinkedIn",
    icon: "/img/LinkedIn.png",
    href: "https://www.linkedin.com/in/liangchao-deng-7b420b269/",
    color: "#0077b5",
  },
  {
    name: "Hugging Face",
    icon: "/img/HuggingFace.png",
    href: "https://huggingface.co/smiler488",
    color: "#ff9d00",
  },
  {
    name: "Bluesky",
    icon: "/img/Bluesky.png",
    href: "https://bsky.app/profile/smiler488.bsky.social",
    color: "#1285fe",
  },
  {
    name: "GitHub",
    icon: "/img/Github.png",
    href: "https://github.com/smiler488",
    color: "#6366f1",
  },
];

const COPY = {
  en: {
    eyebrow: "Research · Software · Collaboration",
    title: "Making plant science measurable with AI.",
    description:
      "Postdoctoral research across plant phenotyping, computer vision, remote sensing, and process-based crop modeling.",
    profile: "View research profile",
    navigation: "Explore",
    navigationLabel: "Footer navigation",
    footerLabel: "Liangchao Deng site footer",
    contact: "Contact",
    community: "Community",
    academic: "Academic",
    collaboration: "Collaboration",
    assistant: "Assistant",
    newWindow: "opens in a new tab",
    privacy: "Research tools are designed with local-first data handling.",
    rights: "All rights reserved.",
  },
  zh: {
    eyebrow: "研究 · 软件 · 合作",
    title: "用人工智能让植物科学可测量。",
    description:
      "围绕作物表型、计算机视觉、遥感与过程驱动作物模型开展博士后研究。",
    profile: "查看研究履历",
    navigation: "站内导航",
    navigationLabel: "页脚导航",
    footerLabel: "邓良超个人网站页脚",
    contact: "联系",
    community: "社区平台",
    academic: "学术邮箱",
    collaboration: "商业合作",
    assistant: "助理邮箱",
    newWindow: "在新标签页中打开",
    privacy: "研究工具优先采用本地数据处理。",
    rights: "保留所有权利。",
  },
};

function SocialIcon({ social, newWindow }) {
  return (
    <Link
      href={social.href}
      target="_blank"
      rel="noopener noreferrer"
      className={styles.socialIcon}
      aria-label={`${social.name} · ${newWindow}`}
      title={social.name}
      style={{ "--social-accent": social.color }}
    >
      <img
        src={social.icon}
        alt=""
        width="26"
        height="26"
        loading="lazy"
        decoding="async"
      />
    </Link>
  );
}

export default function SiteFooter() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? COPY.zh : COPY.en;
  const languageKey = isChinese ? "zh" : "en";
  const year = new Date().getFullYear();

  return (
    <footer className={styles.footer} aria-label={copy.footerLabel}>
      <div className={styles.footerSurface}>
        <div className={styles.footerGrid}>
          <section className={styles.identity}>
            <span className={styles.eyebrow}>{copy.eyebrow}</span>
            <Heading as="h2" className={styles.title}>
              {copy.title}
            </Heading>
            <p className={styles.description}>{copy.description}</p>
            <Link className={styles.profileLink} to="/cv">
              <span>{copy.profile}</span>
              <span aria-hidden="true">→</span>
            </Link>
          </section>

          <nav className={styles.navColumn} aria-label={copy.navigationLabel}>
            <Heading as="h3" className={styles.columnTitle}>
              {copy.navigation}
            </Heading>
            <ul className={styles.linkList}>
              {NAV_LINKS.map((link) => (
                <li key={link.to}>
                  <Link to={link.to}>{link[languageKey]}</Link>
                </li>
              ))}
            </ul>
          </nav>

          <section className={styles.contactColumn}>
            <Heading as="h3" className={styles.columnTitle}>
              {copy.contact}
            </Heading>
            <ul className={styles.contactList}>
              {CONTACT_LINKS.map((contact) => (
                <li key={contact.type}>
                  <Link
                    href={contact.href}
                    target={contact.isExternal ? "_blank" : "_self"}
                    rel={contact.isExternal ? "noopener noreferrer" : undefined}
                  >
                    <span>{copy[contact.type]}</span>
                    <strong>{isChinese ? contact.labelZh : contact.labelEn}</strong>
                  </Link>
                </li>
              ))}
            </ul>
          </section>

          <section className={styles.communityColumn}>
            <Heading as="h3" className={styles.columnTitle}>
              {copy.community}
            </Heading>
            <div className={styles.socialGrid}>
              {SOCIAL_LINKS.map((social) => (
                <SocialIcon
                  key={social.name}
                  social={social}
                  newWindow={copy.newWindow}
                />
              ))}
            </div>
          </section>
        </div>

        <div className={styles.footerBottom}>
          <span>
            © {year} Liangchao Deng. {copy.rights}
          </span>
          <span>{copy.privacy}</span>
        </div>
      </div>
    </footer>
  );
}
