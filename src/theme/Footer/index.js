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
    globeEyebrow: "Global visitors",
    globeTitle: "A research network without borders.",
    globeDescription:
      "This live globe shows the approximate geographic distribution of visitors to this site.",
    globeFrameTitle: "Interactive MapMyVisitors visitor globe",
    globeLoading: "Loading live visitor globe…",
    globeError: "The visitor globe could not be loaded.",
    globeRetry: "Retry",
    globeLink: "Visitor analytics & privacy",
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
    globeEyebrow: "全球访客",
    globeTitle: "让研究连接跨越边界。",
    globeDescription: "实时地球展示访问本网站用户的大致地理分布。",
    globeFrameTitle: "MapMyVisitors 交互式访客地球",
    globeLoading: "正在加载实时访客地球…",
    globeError: "访客地球暂时无法加载。",
    globeRetry: "重新加载",
    globeLink: "访客统计与隐私说明",
    privacy: "本页脚会自动加载 MapMyVisitors，并处理网络与设备元数据。",
    privacyAction: "隐私说明",
    rights: "保留所有权利。",
  },
};

const GLOBE_MESSAGE_SOURCE = "smiler488-mapmyvisitors-globe";
const GLOBE_SCRIPT_SRC =
  "https://mapmyvisitors.com/globe.js?d=q0eg2_fWgmNEh1nVyYkGP7OMwUA7DZIjlDAPMYt-gVI&w=236";

function createGlobeDocument() {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=236, initial-scale=1" />
    <meta name="referrer" content="no-referrer" />
    <base target="_blank" />
    <style>
      html, body { width: 236px; height: 256px; margin: 0; overflow: hidden; background: transparent; }
      body { display: grid; place-items: start center; }
    </style>
  </head>
  <body>
    <script>
      (function () {
        var marker = ${JSON.stringify(GLOBE_MESSAGE_SOURCE)};
        var currentStatus = "loading";
        var parentVisible = false;
        var motionReady = false;
        var observer;
        var poll;
        var timeout;

        function notify(status) {
          currentStatus = status;
          parent.postMessage({ source: marker, type: "status", status: status }, "*");
        }

        function secureLink() {
          var link = document.querySelector("#mmvst_a");
          if (!link) return;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.title = "MapMyVisitors";
          link.setAttribute("aria-label", "View the MapMyVisitors visitor globe");
        }

        function setMotion(visible) {
          if (!motionReady || !window.globe_jq || currentStatus !== "ready") return;
          var shouldMove = visible && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
          var parts = window.globe_jq(".mmvst_globe, .mmvst_map_f, .mmvst_map_b, .mmvst_dots");
          if (!parts.length) return;
          parts.velocity(shouldMove ? "resume" : "pause", true);
        }

        function finish(status) {
          if (currentStatus !== "loading") return;
          window.clearInterval(poll);
          window.clearTimeout(timeout);
          if (observer) observer.disconnect();
          secureLink();
          notify(status);
          if (status === "ready") {
            window.setTimeout(function () {
              motionReady = true;
              setMotion(parentVisible);
            }, 900);
          }
        }

        function checkGlobe() {
          secureLink();
          var inner = document.querySelector(".mmvst_inner");
          var map = document.querySelector(".mmvst_map");
          if (
            inner &&
            map &&
            window.getComputedStyle(inner).display !== "none" &&
            map.offsetWidth > 0 &&
            map.offsetHeight > 0
          ) {
            finish("ready");
          }
        }

        window.addEventListener("message", function (event) {
          if (event.source !== parent || !event.data || event.data.source !== marker) return;
          if (event.data.type === "probe") notify(currentStatus);
          if (event.data.type === "visibility") {
            parentVisible = Boolean(event.data.visible);
            setMotion(parentVisible);
          }
        });

        window.addEventListener("error", function (event) {
          if (event.target && event.target.id === "mmvst_globe") finish("error");
        }, true);

        observer = new MutationObserver(checkGlobe);
        observer.observe(document.documentElement, {
          attributes: true,
          childList: true,
          subtree: true,
        });
        poll = window.setInterval(checkGlobe, 250);
        timeout = window.setTimeout(function () { finish("error"); }, 20000);
        window.addEventListener("load", checkGlobe);
      })();
    </script>
    <script type="text/javascript" id="mmvst_globe" src="${GLOBE_SCRIPT_SRC}" referrerpolicy="no-referrer"></script>
  </body>
</html>`;
}

const GLOBE_DOCUMENT = createGlobeDocument();

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

function FooterGlobe({ copy }) {
  const frameRef = React.useRef(null);
  const stageRef = React.useRef(null);
  const visibleRef = React.useRef(false);
  const [status, setStatus] = React.useState("loading");
  const [attempt, setAttempt] = React.useState(0);

  const postToFrame = React.useCallback((message) => {
    frameRef.current?.contentWindow?.postMessage(
      { source: GLOBE_MESSAGE_SOURCE, ...message },
      "*"
    );
  }, []);

  React.useEffect(() => {
    function handleMessage(event) {
      if (
        event.source !== frameRef.current?.contentWindow ||
        event.data?.source !== GLOBE_MESSAGE_SOURCE ||
        event.data?.type !== "status"
      ) {
        return;
      }

      if (event.data.status === "ready" || event.data.status === "error") {
        setStatus(event.data.status);
      }
    }

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  React.useEffect(() => {
    const stage = stageRef.current;
    if (!stage || typeof IntersectionObserver === "undefined") {
      visibleRef.current = true;
      postToFrame({ type: "visibility", visible: true });
      return undefined;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        visibleRef.current = entry.isIntersecting;
        postToFrame({ type: "visibility", visible: entry.isIntersecting });
      },
      { rootMargin: "240px 0px" }
    );
    observer.observe(stage);
    return () => observer.disconnect();
  }, [attempt, postToFrame]);

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      postToFrame({ type: "probe" });
      postToFrame({ type: "visibility", visible: visibleRef.current });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [attempt, postToFrame]);

  function retry() {
    setStatus("loading");
    setAttempt((current) => current + 1);
  }

  function syncFrame() {
    postToFrame({ type: "probe" });
    postToFrame({ type: "visibility", visible: visibleRef.current });
  }

  return (
    <div
      ref={stageRef}
      className={styles.globeStage}
      data-status={status}
      aria-busy={status === "loading"}
    >
      <iframe
        key={attempt}
        ref={frameRef}
        className={styles.globeFrame}
        title={copy.globeFrameTitle}
        width="236"
        height="256"
        srcDoc={GLOBE_DOCUMENT}
        sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox"
        referrerPolicy="no-referrer"
        onLoad={syncFrame}
      />

      {status === "loading" && (
        <div className={styles.globeLoading} role="status">
          <span aria-hidden="true" />
          {copy.globeLoading}
        </div>
      )}

      {status === "error" && (
        <div className={styles.globeError} role="alert">
          <span>{copy.globeError}</span>
          <button type="button" onClick={retry}>
            {copy.globeRetry}
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
          aria-labelledby="footer-globe-heading"
        >
          <div className={styles.globeBandCopy}>
            <span className={styles.globeEyebrow}>{copy.globeEyebrow}</span>
            <Heading as="h3" id="footer-globe-heading">
              {copy.globeTitle}
            </Heading>
            <p>{copy.globeDescription}</p>
            <Link className={styles.globePrivacyLink} to="/privacy">
              {copy.globeLink}
              <span aria-hidden="true">→</span>
            </Link>
          </div>
          <FooterGlobe copy={copy} />
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
