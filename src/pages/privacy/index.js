import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

const COPY = {
  en: {
    pageTitle: "Privacy & Visitor Analytics",
    pageDescription:
      "How the site-wide visitor globe and MapMyVisitors handle data.",
    eyebrow: "Privacy · Visitor analytics",
    title: "Clear boundaries for the visitor globe.",
    intro:
      "The footer on every page automatically loads a MapMyVisitors globe. This page explains what is sent, why it is used, and what the isolated widget cannot access.",
    updated: "Last updated · July 16, 2026",
    summaryLabel: "Privacy summary",
    summaries: [
      {
        label: "Service",
        value: "MapMyVisitors in a sandboxed frame",
      },
      {
        label: "Purpose",
        value: "Approximate visitor locations and aggregate traffic",
      },
      {
        label: "Privacy contact",
        value: "googalphdlc@gmail.com",
        href: "mailto:googalphdlc@gmail.com",
      },
    ],
    automaticTitle: "Automatic, site-wide loading",
    automaticText:
      "The visitor globe loads without a click whenever a page containing the global footer opens. Its network requests therefore begin during the page visit, even if the footer is not yet visible. The globe animation is paused while it is far outside the viewport, but visitor measurement still loads immediately.",
    dataTitle: "Data MapMyVisitors may process",
    dataIntro:
      "According to the provider’s documentation, the service may receive or infer:",
    dataItems: [
      "IP address and approximate country, region, city, time zone, coordinates, or postal area",
      "Browser, operating system, device characteristics, language, and screen information",
      "The current page and referring URL when the browser makes them available",
      "Visit time, visit count, and session or interaction metadata",
    ],
    referrerNote:
      "The embedded frame uses a no-referrer policy to minimize page-path disclosure in outbound requests. Browser behavior and future provider code can still affect which fields are available.",
    useTitle: "Purpose, recipient, and retention",
    useText:
      "This site uses the information to render the live globe, understand broad geographic visitor origins, and review aggregate traffic. Requests go directly to MapMyVisitors. This site does not receive your precise GPS location and does not sell visitor data. Provider-side storage, retention, deletion, and rights requests are governed by MapMyVisitors policies.",
    isolationTitle: "What the globe cannot access",
    isolationText:
      "The legacy widget runs inside an opaque sandbox without same-origin access or camera permission. It cannot directly read the parent page DOM, browser-stored AI keys, camera streams, or research files uploaded to site tools. It can access only its own frame and the network requests needed for the globe.",
    localFirstTitle: "Local-first tools remain a separate boundary",
    localFirstText:
      "Many research tools process data locally in the browser, while optional AI features send data only to a provider chosen by the user. Those tool-specific boundaries are described in their interfaces and tutorials. “Local-first” does not mean that the surrounding page makes no third-party request: the footer globe is the disclosed exception.",
    cookiesTitle: "Session identifiers and browser controls",
    cookiesText:
      "The provider states that it does not use cross-site tracking cookies, although service responses may attempt to set a technical session identifier. Whether it is accepted depends on browser and third-party cookie settings. Browser content blocking, DNS filtering, or script blocking can prevent the MapMyVisitors request, in which case the globe may show an error.",
    choiceTitle: "Consent and regional requirements",
    choiceText:
      "Some jurisdictions may require consent or another lawful basis before third-party visitor analytics loads. This notice describes the automatic behavior but is not itself a consent mechanism. Questions or rights requests can be sent to the privacy contact above or directly to MapMyVisitors under its policy.",
    externalTitle: "Provider documents",
    externalText:
      "Read the provider’s current documents for its processing, retention, and contact details.",
    policyAction: "MapMyVisitors privacy policy",
    termsAction: "MapMyVisitors terms of service",
    backAction: "Return home",
  },
  zh: {
    pageTitle: "隐私与访客统计",
    pageDescription: "说明全站访客地球及 MapMyVisitors 如何处理数据。",
    eyebrow: "隐私 · 访客统计",
    title: "明确访客地球的数据边界。",
    intro:
      "每个页面的底栏都会自动加载 MapMyVisitors 访客地球。本页说明会发送哪些信息、使用目的，以及隔离挂件无法访问的内容。",
    updated: "最后更新 · 2026 年 7 月 16 日",
    summaryLabel: "隐私摘要",
    summaries: [
      {
        label: "第三方服务",
        value: "沙箱隔离的 MapMyVisitors",
      },
      {
        label: "使用目的",
        value: "访客大致位置与汇总访问统计",
      },
      {
        label: "隐私联系",
        value: "googalphdlc@gmail.com",
        href: "mailto:googalphdlc@gmail.com",
      },
    ],
    automaticTitle: "全站自动加载",
    automaticText:
      "只要打开包含全局底栏的页面，访客地球就会自动加载，无需点击。因此，即使底栏尚未出现在屏幕中，相关网络请求也会在访问期间开始。地球远离可视区域时会暂停旋转动画，但访客统计仍会立即加载。",
    dataTitle: "MapMyVisitors 可能处理的数据",
    dataIntro: "根据服务提供方的说明，该服务可能接收或推断：",
    dataItems: [
      "IP 地址，以及大致国家、地区、城市、时区、经纬度或邮政区域",
      "浏览器、操作系统、设备特征、语言与屏幕信息",
      "浏览器允许提供时的当前页面和来源网址",
      "访问时间、访问次数，以及会话或交互元数据",
    ],
    referrerNote:
      "隔离框架采用 no-referrer 策略，以尽量减少向外部请求披露具体页面路径；浏览器行为和服务方后续代码仍可能影响可获得的字段。",
    useTitle: "用途、数据接收方与保存",
    useText:
      "本网站使用这些信息展示实时地球、了解访客的大致地理来源并查看汇总流量。请求会直接发送给 MapMyVisitors。本网站不会获得你的精确 GPS 位置，也不会出售访客数据。服务方的数据保存、删除和权利请求以 MapMyVisitors 的政策为准。",
    isolationTitle: "访客地球无法访问的内容",
    isolationText:
      "旧版挂件运行在不具备同源权限和摄像头权限的不透明沙箱中，无法直接读取父页面 DOM、浏览器中保存的 AI 密钥、摄像头画面或上传到科研工具的文件；它只能访问自身框架和地球运行所需的网络请求。",
    localFirstTitle: "本地优先工具具有独立的数据边界",
    localFirstText:
      "许多科研工具会在浏览器本地处理数据；可选 AI 功能仅向用户自行选择的服务商发送数据，具体边界会在相应界面和教程中说明。“本地优先”并不代表页面完全没有第三方请求：底栏访客地球是这里明确披露的例外。",
    cookiesTitle: "会话标识与浏览器控制",
    cookiesText:
      "服务方声明不会使用跨站跟踪 Cookie，但服务响应仍可能尝试设置技术性会话标识；是否接受取决于浏览器和第三方 Cookie 设置。浏览器内容拦截、DNS 过滤或脚本拦截可以阻止 MapMyVisitors 请求，此时地球可能显示加载失败。",
    choiceTitle: "同意与地区要求",
    choiceText:
      "部分司法辖区可能要求在第三方访客统计加载前取得同意或具备其他合法依据。本说明披露自动加载行为，但本身不是同意机制。如有问题或权利请求，可联系上述隐私邮箱，或按照 MapMyVisitors 政策直接联系服务方。",
    externalTitle: "服务方文件",
    externalText: "请查阅服务方的最新文件，了解其数据处理、保存与联系方式。",
    policyAction: "MapMyVisitors 隐私政策",
    termsAction: "MapMyVisitors 服务条款",
    backAction: "返回首页",
  },
};

function SummaryItem({ item }) {
  return (
    <article className={styles.summaryItem}>
      <span>{item.label}</span>
      {item.href ? (
        <Link href={item.href}>{item.value}</Link>
      ) : (
        <strong>{item.value}</strong>
      )}
    </article>
  );
}

export default function PrivacyPage() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? COPY.zh : COPY.en;

  return (
    <Layout title={copy.pageTitle} description={copy.pageDescription}>
      <main className={styles.page}>
        <header className={styles.hero}>
          <span className={styles.eyebrow}>{copy.eyebrow}</span>
          <Heading as="h1">{copy.title}</Heading>
          <p>{copy.intro}</p>
          <span className={styles.updated}>{copy.updated}</span>
        </header>

        <section className={styles.summaryGrid} aria-label={copy.summaryLabel}>
          {copy.summaries.map((item) => (
            <SummaryItem key={item.label} item={item} />
          ))}
        </section>

        <div className={styles.contentGrid}>
          <article className={styles.contentCard}>
            <Heading as="h2">{copy.automaticTitle}</Heading>
            <p>{copy.automaticText}</p>
          </article>

          <article className={`${styles.contentCard} ${styles.dataCard}`}>
            <Heading as="h2">{copy.dataTitle}</Heading>
            <p>{copy.dataIntro}</p>
            <ul>
              {copy.dataItems.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
            <p className={styles.note}>{copy.referrerNote}</p>
          </article>

          <article className={styles.contentCard}>
            <Heading as="h2">{copy.useTitle}</Heading>
            <p>{copy.useText}</p>
          </article>

          <article className={styles.contentCard}>
            <Heading as="h2">{copy.isolationTitle}</Heading>
            <p>{copy.isolationText}</p>
          </article>

          <article className={styles.contentCard}>
            <Heading as="h2">{copy.localFirstTitle}</Heading>
            <p>{copy.localFirstText}</p>
          </article>

          <article className={styles.contentCard}>
            <Heading as="h2">{copy.cookiesTitle}</Heading>
            <p>{copy.cookiesText}</p>
          </article>
        </div>

        <aside className={styles.consentNotice}>
          <Heading as="h2">{copy.choiceTitle}</Heading>
          <p>{copy.choiceText}</p>
        </aside>

        <section className={styles.providerCard}>
          <div>
            <Heading as="h2">{copy.externalTitle}</Heading>
            <p>{copy.externalText}</p>
          </div>
          <div className={styles.providerLinks}>
            <Link
              href="https://mapmyvisitors.com/b/policy"
              target="_blank"
              rel="noopener noreferrer"
            >
              {copy.policyAction}
              <span aria-hidden="true">↗</span>
            </Link>
            <Link
              href="https://mapmyvisitors.com/b/tos"
              target="_blank"
              rel="noopener noreferrer"
            >
              {copy.termsAction}
              <span aria-hidden="true">↗</span>
            </Link>
          </div>
        </section>

        <Link className={styles.backLink} to="/">
          <span aria-hidden="true">←</span>
          {copy.backAction}
        </Link>
      </main>
    </Layout>
  );
}
