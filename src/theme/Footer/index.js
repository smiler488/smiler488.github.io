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
    email: "googalphdlc@gmail.com",
  },
  {
    type: "collaboration",
    email: "dengliangchao@smiler488.com",
  },
  {
    type: "assistant",
    email: "smiler488@agent.qq.com",
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
    visitorMapEyebrow: "Global visitors",
    visitorMapTitle: "A research network without borders.",
    visitorMapDescription:
      "This live map shows the approximate geographic distribution of visitors to this site.",
    visitorMapFrameTitle: "MapMyVisitors visitor map",
    visitorMapLoading: "Loading live visitor map…",
    visitorMapError: "The visitor map could not be loaded.",
    visitorMapRetry: "Retry",
    visitorMapLink: "Visitor analytics & privacy",
    privacy:
      "This footer automatically loads MapMyVisitors and processes network and device metadata.",
    privacyAction: "Privacy",
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
    visitorMapEyebrow: "全球访客",
    visitorMapTitle: "让研究连接跨越边界。",
    visitorMapDescription: "实时地图展示访问本网站用户的大致地理分布。",
    visitorMapFrameTitle: "MapMyVisitors 访客地图",
    visitorMapLoading: "正在加载实时访客地图…",
    visitorMapError: "访客地图暂时无法加载。",
    visitorMapRetry: "重新加载",
    visitorMapLink: "访客统计与隐私说明",
    privacy: "本页脚会自动加载 MapMyVisitors，并处理网络与设备元数据。",
    privacyAction: "隐私说明",
    rights: "保留所有权利。",
  },
};

const MAP_MESSAGE_SOURCE = "smiler488-mapmyvisitors-map";
const MAP_PROFILE_URL = "https://mapmyvisitors.com/web/1c0ty";
const MAP_IMAGE_SRC =
  "https://mapmyvisitors.com/map.png?d=ccC5JZBvNNpRHfn94y3CRXzvvcSb99CMKXuy-7wzczI&cl=ffffff&w=360";

function createVisitorMapDocument() {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="referrer" content="no-referrer" />
    <base target="_blank" />
    <style>
      *, *::before, *::after { box-sizing: border-box; }
      html, body { width: 100%; height: 212px; margin: 0; overflow: hidden; background: transparent; }
      body { display: grid; place-items: center; }
      #map-image-link {
        display: grid;
        width: 100%;
        height: 100%;
        place-items: center;
        overflow: hidden;
        border-radius: 14px;
      }
      #map-image {
        display: block;
        width: 100%;
        max-width: 360px;
        height: auto;
        max-height: 212px;
        object-fit: contain;
      }
    </style>
  </head>
  <body>
    <a
      id="map-image-link"
      href="${MAP_PROFILE_URL}"
      title="View visitor statistics"
      aria-label="View MapMyVisitors visitor statistics"
      target="_blank"
      rel="noopener noreferrer"
    >
      <img
        id="map-image"
        src="${MAP_IMAGE_SRC}"
        alt="Approximate locations of website visitors"
        width="360"
        height="199"
      />
    </a>
    <script>
      (function () {
        var marker = ${JSON.stringify(MAP_MESSAGE_SOURCE)};
        var image = document.querySelector("#map-image");
        var currentStatus = "loading";

        function notify(status) {
          currentStatus = status;
          parent.postMessage(
            { source: marker, type: "status", status: status, mode: "image" },
            "*"
          );
        }

        function reportImageStatus() {
          if (currentStatus !== "loading") return;
          notify(image && image.naturalWidth > 0 ? "ready" : "error");
        }

        window.addEventListener("message", function (event) {
          if (event.source !== parent || !event.data || event.data.source !== marker) return;
          if (event.data.type === "probe") notify(currentStatus);
        });

        if (!image) {
          notify("error");
          return;
        }

        image.addEventListener("load", reportImageStatus, { once: true });
        image.addEventListener("error", reportImageStatus, { once: true });
        if (image.complete) window.setTimeout(reportImageStatus, 0);
      })();
    </script>
  </body>
</html>`;
}

const MAP_DOCUMENT = createVisitorMapDocument();

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

function FooterVisitorMap({ copy }) {
  const frameRef = React.useRef(null);
  const [status, setStatus] = React.useState("loading");
  const [renderMode, setRenderMode] = React.useState("pending");
  const [attempt, setAttempt] = React.useState(0);

  const postToFrame = React.useCallback((message) => {
    frameRef.current?.contentWindow?.postMessage(
      { source: MAP_MESSAGE_SOURCE, ...message },
      "*"
    );
  }, []);

  React.useEffect(() => {
    function handleMessage(event) {
      if (
        event.source !== frameRef.current?.contentWindow ||
        event.data?.source !== MAP_MESSAGE_SOURCE ||
        event.data?.type !== "status"
      ) {
        return;
      }

      if (event.data.status === "ready" || event.data.status === "error") {
        setStatus(event.data.status);
        setRenderMode(event.data.mode || "unknown");
      }
    }

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      postToFrame({ type: "probe" });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [attempt, postToFrame]);

  React.useEffect(() => {
    if (status !== "loading") return undefined;

    const watchdog = window.setTimeout(() => {
      setStatus("error");
    }, 18000);

    return () => window.clearTimeout(watchdog);
  }, [attempt, status]);

  function retry() {
    setStatus("loading");
    setRenderMode("pending");
    setAttempt((current) => current + 1);
  }

  function syncFrame() {
    postToFrame({ type: "probe" });
  }

  return (
    <div
      className={styles.globeStage}
      data-status={status}
      data-mode={renderMode}
      aria-busy={status === "loading"}
    >
      <iframe
        key={attempt}
        ref={frameRef}
        className={styles.globeFrame}
        title={copy.visitorMapFrameTitle}
        width="360"
        height="212"
        loading="eager"
        srcDoc={MAP_DOCUMENT}
        sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox"
        referrerPolicy="no-referrer"
        onLoad={syncFrame}
      />

      {status === "loading" && (
        <div className={styles.globeLoading} role="status">
          <span aria-hidden="true" />
          {copy.visitorMapLoading}
        </div>
      )}

      {status === "error" && (
        <div className={styles.globeError} role="alert">
          <span>{copy.visitorMapError}</span>
          <button type="button" onClick={retry}>
            {copy.visitorMapRetry}
          </button>
        </div>
      )}
    </div>
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
                <li key={contact.email}>
                  <Link href={`mailto:${contact.email}`} target="_self">
                    <span>{copy[contact.type]}</span>
                    <strong>{contact.email}</strong>
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

        <section
          className={styles.globeBand}
          aria-labelledby="footer-visitor-map-heading"
        >
          <div className={styles.globeBandCopy}>
            <span className={styles.globeEyebrow}>
              {copy.visitorMapEyebrow}
            </span>
            <Heading as="h3" id="footer-visitor-map-heading">
              {copy.visitorMapTitle}
            </Heading>
            <p>{copy.visitorMapDescription}</p>
            <Link className={styles.globePrivacyLink} to="/privacy">
              {copy.visitorMapLink}
              <span aria-hidden="true">→</span>
            </Link>
          </div>
          <FooterVisitorMap copy={copy} />
        </section>

        <div className={styles.footerBottom}>
          <span>
            © {year} Liangchao Deng. {copy.rights}
          </span>
          <span className={styles.privacyDisclosure}>
            {copy.privacy} <Link to="/privacy">{copy.privacyAction}</Link>
          </span>
        </div>
      </div>
    </footer>
  );
}
