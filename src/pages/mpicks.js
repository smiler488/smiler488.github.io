import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import styles from "./mpicks.module.css";

const PICKS_DATA = {
  en: {
    pageTitle: "mPicks",
    pageDescription: "Curated tools, cloud resources, and AI services optimized for global academic research and productivity.",
    title: "mPicks",
    subtitle: "Recommended tools, services, and hardware optimized for researchers and developers.",
    openBtn: "Visit Site",
    items: [
      {
        id: "wgetcloud",
        title: "WgetCloud Premium SLA VPN",
        description: "High-availability SLA lines. SLA private line service for seamless, ultra-fast access to global AI assistants (Claude, ChatGPT, Gemini) and media. Professional and business first choice.",
        link: "https://invite.wgetcloud.ltd/auth/register?code=Y0qXc0",
        badges: ["SLA Private Line", "Ultra Fast AI", "All Platforms"],
        tint: "blue",
      },
      {
        id: "dgcloud",
        title: "DG Cloud Budget Any-Protocol VPN",
        description: "Global exclusive Any protocol for high-performance and budget-friendly acceleration of dev workflows, academic downloads, and HD videos. Cross-platform support.",
        link: "https://inv.dginv.click/#/register?code=BcP8L3n2",
        badges: ["Budget Choice", "Any Protocol", "Dev Workflow"],
        tint: "green",
      },
      {
        id: "xiaomimimo",
        title: "Xiaomi MiMo Open Platform",
        description: "Experience Xiaomi's top multimodal large-scale model MiMo V2.5. Use my link to sign up to get ¥10 API trial credit and 10% off your first order. High性价比 and top-tier capabilities.",
        link: "https://platform.xiaomimimo.com?ref=E9434S",
        badges: ["Multimodal AI", "Xiaomi V2.5", "¥10 Trial Gift"],
        tint: "violet",
      },
      {
        id: "tencentcloud",
        title: "Tencent Cloud Server & TokenHub",
        description: "2C2G4M high-speed cloud servers from ¥99/year. Minutes deployment for OpenClaw, Hermes, DeepSeek. TokenHub integrates Hunyuan and third-party models, enabling speech, vision, and 3D generation.",
        link: "https://curl.qcloud.com/M9Y6HIFv",
        badges: ["¥99 Server", "AI Host Ready", "TokenHub Hub"],
        tint: "orange",
      },
    ],
  },
  zh: {
    pageTitle: "好物推荐",
    pageDescription: "精选学术加速、大模型计算与云服务好物，助力高效科研开发与数字工作流。",
    title: "好物推荐",
    subtitle: "精选学术加速、大模型计算与云服务好物，助力高效科研开发与数字工作流。",
    openBtn: "立即访问",
    items: [
      {
        id: "wgetcloud",
        title: "WgetCloud 精品专线网络加速",
        description: "极致体验，全平台通用，商务与专业首选。SLA 专线高可用，稳定支持全球流媒体 + AI（快速流畅访问 Claude、ChatGPT、Gemini 等）。",
        link: "https://invite.wgetcloud.ltd/auth/register?code=Y0qXc0",
        badges: ["精品专线", "AI 极速访问", "SLA 高可用"],
        tint: "blue",
      },
      {
        id: "dgcloud",
        title: "DG Cloud 平价专线网络加速",
        description: "全球独家 Any 协议，稳定加速全球工作流与超清视频，全平台通用。高性价比的科研加速之选。",
        link: "https://inv.dginv.click/#/register?code=BcP8L3n2",
        badges: ["平价首选", "Any 协议", "工作流加速"],
        tint: "green",
      },
      {
        id: "xiaomimimo",
        title: "小米 MiMo 大模型开放平台",
        description: "体验小米顶尖多模态大模型 MiMo V2.5。通过邀请链接注册，双方各得 ¥10 API 体验金（40天有效）+ 首单 9 折。体验无限 AI 智能可能。",
        link: "https://platform.xiaomimimo.com?ref=E9434S",
        badges: ["多模态 AI", "小米顶尖模型", "送 10元 体验金"],
        tint: "violet",
      },
      {
        id: "tencentcloud",
        title: "腾讯云轻量服务器 & TokenHub",
        description: "新客特惠 2核2G4M 服务器 99元/年起。分钟级部署 OpenClaw、Hermes 或 DeepSeek-TUI。TokenHub 统一大模型入口，轻松实现多模态及图生三维。",
        link: "https://curl.qcloud.com/M9Y6HIFv",
        badges: ["99元/年起", "AI 智能体专属", "混元多模态"],
        tint: "orange",
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
        </header>

        <div className={styles.grid}>
          {copy.items.map((item) => (
            <div key={item.id} className={styles.card} data-tint={item.tint}>
              <div className={styles.cardInner}>
                <div className={styles.cardHeader}>
                  <div className={styles.iconWrapper}>
                    {item.tint === "blue" && (
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                      </svg>
                    )}
                    {item.tint === "green" && (
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                      </svg>
                    )}
                    {item.tint === "violet" && (
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                      </svg>
                    )}
                    {item.tint === "orange" && (
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
                        <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
                        <line x1="6" y1="6" x2="6.01" y2="6" />
                        <line x1="6" y1="18" x2="6.01" y2="18" />
                      </svg>
                    )}
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

                <Link
                  className={styles.cardLink}
                  to={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span>{copy.openBtn}</span>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="7" y1="17" x2="17" y2="7" />
                    <polyline points="7 7 17 7 17 17" />
                  </svg>
                </Link>
              </div>
            </div>
          ))}
        </div>
      </main>
    </Layout>
  );
}
