import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import styles from "./mpicks.module.css";

function getFaviconUrl(url) {
  try {
    const hostname = new URL(url).hostname;
    return `https://www.google.com/s2/favicons?domain=${hostname}&sz=64`;
  } catch {
    return null;
  }
}

const PICKS_DATA = {
  en: {
    pageTitle: "mPicks",
    pageDescription:
      "Curated tools, cloud resources, and AI services optimized for global academic research and productivity.",
    title: "mPicks",
    subtitle:
      "Recommended tools, services, and hardware optimized for researchers and developers.",
    openBtn: "Visit Site",
    qrScanHint: "Scan QR to visit",
    disclosure:
      "Links on this page contain invite codes. Signing up through them may earn the author a referral reward.",
    items: [
      {
        id: "wgetcloud",
        title: "WgetCloud Premium SLA VPN",
        description:
          "Premium SLA high-availability private line service for seamless, ultra-fast access to global streaming and AI assistants (Claude, ChatGPT, Gemini). The business and professional first choice.",
        link: "https://invite.wgetcloud.ltd/auth/register?code=Y0qXc0",
        badges: ["SLA Private Line", "Ultra Fast AI", "All Platforms"],
        tint: "blue",
      },
      {
        id: "dgcloud",
        title: "DG Cloud Budget Any-Protocol VPN",
        description:
          "Global exclusive Any protocol for high-performance and budget-friendly acceleration of dev workflows, academic downloads, and HD videos. Cross-platform universal support.",
        link: "https://inv.dginv.click/#/register?code=BcP8L3n2",
        badges: ["Budget Choice", "Any Protocol", "Dev Workflow"],
        tint: "green",
      },
      {
        id: "xiaomimimo",
        title: "Xiaomi MiMo Open Platform",
        description:
          "Experience Xiaomi's top multimodal large-scale model MiMo V2.5. New accounts get ¥10 API trial credit and 10% off the first order (code applied automatically, credit valid for 40 days). High-performance AI capabilities.",
        link: "https://platform.xiaomimimo.com?ref=E9434S",
        badges: ["Multimodal AI", "Xiaomi V2.5", "¥10 Trial Credit"],
        tint: "violet",
      },
      {
        id: "tencentcloud",
        title: "Tencent Cloud Server & TokenHub",
        description:
          "2C2G4M servers from ¥99/year. Deployed on overseas servers, mainland China users can safely command their OpenClaw, Hermes, or DeepSeek-TUI agents via WeChat/Lark to safely retrieve overseas web resources and data. Features TokenHub for speech, reasoning, and 3D generation.",
        link: "https://curl.qcloud.com/M9Y6HIFv",
        badges: ["¥99 Server", "AI Host Ready", "TokenHub Hub"],
        tint: "orange",
      },
      {
        id: "qoder",
        title: "Qoder AI Coding Agent",
        description:
          "An agentic development platform from Tongyi Yunqi (Alibaba). Multi-agent coordination spans a desktop IDE, a terminal CLI and cloud agents that plan, execute and deliver coding tasks, with additional workflows for legal, finance and HR work.",
        link: "https://qoder.com.cn/referral?referral_code=3oqWGfbKqMVzpHxk0XVmw7h21J4XOU4R",
        badges: ["Agentic IDE", "Terminal CLI", "Cloud Agents"],
        tint: "violet",
      },
      {
        id: "kimi",
        title: "Kimi AI Assistant (Moonshot AI)",
        description:
          "Moonshot AI's Kimi assistant, built on the K-series long-context models for literature reading, long-form writing and agentic coding. New accounts get a 7-day membership credit; log in to Kimi shortly after signing up to claim it.",
        link: "https://kimi-bot.com/activities/viral-referral/share?scenario=invite&from=share_poster&invitation_code=SDZGYW",
        badges: ["Long Context", "Agentic Coding", "7-Day Credit"],
        tint: "green",
      },
      {
        id: "ima",
        title: "ima Tencent AI Knowledge Assistant",
        description:
          "Tencent's ima builds a personal knowledge base from your own documents and lets you ask questions across it, with desktop and mobile clients. The link opens a limited-time invite campaign for compute credits.",
        link: "https://ima.qq.com/copilot-invite-reward-token/assist/V_5sH_zUlOz6Vxsf86bOyw",
        badges: ["Knowledge Base", "Tencent AI", "Desktop + Mobile"],
        tint: "orange",
      },
      {
        id: "aliyun",
        title: "Alibaba Cloud Deals Mini-site",
        description:
          "Alibaba Cloud's discount mini-site, covering LLM token plans, cloud servers, storage and AI developer tools. A practical entry point when you need hosting or model API credits for a project.",
        link: "https://www.aliyun.com/minisite/goods?userCode=f4o3ca77",
        badges: ["Token Plans", "Cloud Servers", "New-user Offers"],
        tint: "blue",
      },
      {
        id: "workbuddy",
        title: "WorkBuddy AI Work Assistant",
        description:
          "An AI-powered work assistant that helps you manage tasks, schedule meetings, and boost productivity. Sign up via the invite link to get started with exclusive benefits.",
        link: "https://www.workbuddy.cn/events/invite?inviteCode=uxw36fola8",
        badges: ["AI Assistant", "Task Management", "Invite Benefits"],
        tint: "green",
      },
    ],
  },
  zh: {
    pageTitle: "好物推荐",
    pageDescription:
      "精选学术加速、大模型计算与云服务好物，助力高效科研开发与数字工作流。",
    title: "好物推荐",
    subtitle:
      "精选学术加速、大模型计算与云服务好物，助力高效科研开发与数字工作流。",
    openBtn: "立即访问",
    qrScanHint: "手机扫码访问",
    disclosure: "本页链接包含邀请码，通过它们注册作者可能获得推广奖励。",
    items: [
      {
        id: "wgetcloud",
        title: "WgetCloud 精品专线网络加速",
        description:
          "精品专线服务，极致体验，全平台通用，商务与专业首选。SLA 专线高可用，稳定支持全球流媒体与 AI 工作流，快速稳定访问 Claude、ChatGPT、Gemini 等主流智能体。",
        link: "https://invite.wgetcloud.ltd/auth/register?code=Y0qXc0",
        badges: ["精品专线", "AI 极速访问", "SLA 高可用"],
        tint: "blue",
      },
      {
        id: "dgcloud",
        title: "DG Cloud 平价专线网络加速",
        description:
          "全球独家 Any 协议网络加速，稳定加速全球工作流与超清视频，全平台通用。高性价比的平价专线加速首选。",
        link: "https://inv.dginv.click/#/register?code=BcP8L3n2",
        badges: ["平价首选", "Any 协议", "工作流加速"],
        tint: "green",
      },
      {
        id: "xiaomimimo",
        title: "小米 MiMo 大模型开放平台",
        description:
          "体验小米顶尖多模态大模型 MiMo V2.5。新用户注册可得 ¥10 API 体验金（首单 9 折，注册后自动填入邀请码，体验金 40 天有效），覆盖对话、逻辑与图像生成等多模态高性价比场景。",
        link: "https://platform.xiaomimimo.com?ref=E9434S",
        badges: ["多模态 AI", "小米顶尖模型", "送 10元 体验金"],
        tint: "violet",
      },
      {
        id: "tencentcloud",
        title: "腾讯云轻量服务器 & TokenHub",
        description:
          "新客特惠 2核2G4M 服务器 99元/年起，支持国内和境外的轻应用服务器。可分钟级部署 OpenClaw / Hermes / DeepSeek - TUI 等智能体，部署在境外服务器上，中国大陆用户即可安全地通过微信/飞书等 IM 软件，向智能体下发指令，从而无感安全地获取境外网络上的资料和信息。同时内置大模型服务 TokenHub 覆盖对话、推理、图生三维等场景。",
        link: "https://curl.qcloud.com/M9Y6HIFv",
        badges: ["99元/年起", "AI 智能体专属", "混元多模态"],
        tint: "orange",
      },
      {
        id: "qoder",
        title: "Qoder AI 编程智能体",
        description:
          "通义云启（阿里）推出的智能体开发平台。以多智能体协同贯通桌面 IDE、终端 CLI 与云端智能体，可自主规划、执行并交付编码任务，同时覆盖法务、财务、人力等专业工作场景。",
        link: "https://qoder.com.cn/referral?referral_code=3oqWGfbKqMVzpHxk0XVmw7h21J4XOU4R",
        badges: ["智能体 IDE", "终端 CLI", "云端智能体"],
        tint: "violet",
      },
      {
        id: "kimi",
        title: "Kimi 智能助手（月之暗面）",
        description:
          "月之暗面 Kimi 智能助手，基于 K 系列长上下文模型，适合文献阅读、长文写作与智能体编程。新用户注册可获得 7 天会员额度，注册后请尽快登录 Kimi 领取。",
        link: "https://kimi-bot.com/activities/viral-referral/share?scenario=invite&from=share_poster&invitation_code=SDZGYW",
        badges: ["超长上下文", "智能体编程", "7 天会员额度"],
        tint: "green",
      },
      {
        id: "ima",
        title: "ima 腾讯 AI 知识管家",
        description:
          "腾讯出品的 ima，可把自己的文档整理成个人知识库并跨库提问，桌面端与移动端均可使用。该链接为限时邀请活动，用于领取算力福利。",
        link: "https://ima.qq.com/copilot-invite-reward-token/assist/V_5sH_zUlOz6Vxsf86bOyw",
        badges: ["个人知识库", "腾讯出品", "桌面 + 移动"],
        tint: "orange",
      },
      {
        id: "aliyun",
        title: "阿里云云小站",
        description:
          "阿里云优惠专区，覆盖大模型 Token 计划、云服务器、存储与 AI 开发工具。当项目需要托管资源或模型 API 额度时，是一个实用的入口。",
        link: "https://www.aliyun.com/minisite/goods?userCode=f4o3ca77",
        badges: ["Token 计划", "云服务器", "新人优惠"],
        tint: "blue",
      },
      {
        id: "workbuddy",
        title: "WorkBuddy AI 工作助手",
        description:
          "AI 驱动的工作助手，帮助你管理任务、安排会议、提升工作效率。通过邀请链接注册可享受专属福利。",
        link: "https://www.workbuddy.cn/events/invite?inviteCode=uxw36fola8",
        badges: ["AI 助手", "任务管理", "邀请福利"],
        tint: "green",
      },
    ],
  },
};

export default function MPicksPage() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? PICKS_DATA.zh : PICKS_DATA.en;

  return (
    <Layout title={copy.pageTitle} description={copy.pageDescription}>
      <main className={styles.container}>
        <header className={styles.header}>
          <div className={styles.headerGlow} />
          <span className={styles.eyebrow}>mPicks</span>
          <Heading as="h1" className={styles.title}>
            {copy.title}
          </Heading>
          <p className={styles.subtitle}>{copy.subtitle}</p>
          <p className={styles.disclosure}>{copy.disclosure}</p>
        </header>

        <div className={styles.grid}>
          {copy.items.map((item) => {
            const faviconUrl = getFaviconUrl(item.link);
            return (
            <div key={item.id} className={styles.card} data-tint={item.tint}>
              <div className={styles.cardInner}>
                <div className={styles.cardHeader}>
                  <div className={styles.iconWrapper}>
                    {faviconUrl ? (
                      <img
                        src={faviconUrl}
                        alt=""
                        width="28"
                        height="28"
                        loading="lazy"
                        className={styles.cardFavicon}
                        onError={(e) => {
                          e.target.style.display = "none";
                          const parent = e.target.closest(".iconWrapper") || e.target.parentElement;
                          if (parent) {
                            const fallback = parent.querySelector(`.${styles.iconSvg}`);
                            if (fallback) fallback.style.display = "block";
                          }
                        }}
                      />
                    ) : null}
                    <span
                      className={styles.iconSvg}
                      style={faviconUrl ? { display: "none" } : undefined}
                    >
                      {item.tint === "blue" && (
                        <svg
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                      )}
                      {item.tint === "green" && (
                        <svg
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                        </svg>
                      )}
                      {item.tint === "violet" && (
                        <svg
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                        </svg>
                      )}
                      {item.tint === "orange" && (
                        <svg
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
                          <rect
                            x="2"
                            y="14"
                            width="20"
                            height="8"
                            rx="2"
                            ry="2"
                          />
                          <line x1="6" y1="6" x2="6.01" y2="6" />
                          <line x1="6" y1="18" x2="6.01" y2="18" />
                        </svg>
                      )}
                    </span>
                  </div>
                  <Heading as="h2" className={styles.cardTitle}>
                    {item.title}
                  </Heading>
                </div>

                <p className={styles.cardDescription}>{item.description}</p>

                <div className={styles.badgeRow}>
                  {item.badges.map((badge) => (
                    <span key={badge} className={styles.badge}>
                      {badge}
                    </span>
                  ))}
                </div>

                <div className={styles.actionGroup}>
                  <Link
                    className={styles.cardLink}
                    to={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <span>{copy.openBtn}</span>
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <line x1="7" y1="17" x2="17" y2="7" />
                      <polyline points="7 7 17 7 17 17" />
                    </svg>
                  </Link>

                  <div className={styles.qrContainer}>
                    <button
                      className={styles.qrButton}
                      aria-label="Show QR Code"
                    >
                      <svg
                        width="20"
                        height="20"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <rect x="3" y="3" width="7" height="7" />
                        <rect x="14" y="3" width="7" height="7" />
                        <rect x="14" y="14" width="7" height="7" />
                        <rect x="3" y="14" width="7" height="7" />
                        <line x1="7" y1="7" x2="7.01" y2="7" />
                        <line x1="17" y1="7" x2="17.01" y2="7" />
                        <line x1="17" y1="17" x2="17.01" y2="17" />
                        <line x1="7" y1="17" x2="7.01" y2="17" />
                      </svg>
                    </button>
                    <div className={styles.qrTooltip} data-particle-obstacle>
                      <img
                        src={`https://api.qrserver.com/v1/create-qr-code/?size=130x130&data=${encodeURIComponent(
                          item.link
                        )}`}
                        alt="QR Code"
                        width="130"
                        height="130"
                        loading="lazy"
                      />
                      <span className={styles.qrText}>{copy.qrScanHint}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );})}
        </div>
      </main>
    </Layout>
  );
}
