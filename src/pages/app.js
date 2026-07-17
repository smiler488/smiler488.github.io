import React, { useMemo, useState, useRef, useEffect } from "react";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import CitationNotice from "../components/CitationNotice";
import { APP_CATEGORIES, APP_MANIFEST, localizeApp } from "../data/appManifest";
import styles from "./app.module.css";

const COPY = {
  en: {
    pageTitle: "App Lab — Free Browser Tools for Plant Science and AI",
    pageDescription:
      "Fourteen free browser tools for field data, crop research, imaging, visualization and AI-assisted workflows. Nothing to install.",
    eyebrow: "Digital plant phenotyping platform",
    title: "A focused lab for field data, imaging and AI.",
    subtitle:
      "Fourteen practical browser tools shaped around plant science workflows. Each workspace now shares one calm, responsive interface while keeping its specialist controls close at hand.",
    statsLabel: "App Lab overview",
    statTools: "browser tools",
    statAreas: "workflow areas",
    statByok: "AI model choice",
    catalogEyebrow: "Explore the toolkit",
    catalogTitle: "Choose a workflow",
    catalogText:
      "Search by task or filter by research stage. Every card opens a dedicated workspace.",
    searchLabel: "Search tools",
    searchPlaceholder: "Search tools or tasks",
    clearSearch: "Clear search",
    filterLabel: "Filter tools by category",
    tool: "tool",
    tools: "tools",
    emptyTitle: "No matching tool",
    emptyText: "Try a broader term or return to the complete App Lab.",
    showAll: "Show all tools",
    open: "Open",
    openTool: "Open tool",
    capabilities: "Capabilities",
    trustLabel: "Privacy and runtime notes",
    trust: [
      {
        title: "Local-first where possible",
        text: "Files and calculations stay in the browser unless a tool clearly names an external service.",
      },
      {
        title: "Your model, your key",
        text: "AI tools use the provider and API credentials you configure for that session.",
      },
      {
        title: "Research-aware output",
        text: "Scientific assumptions, external runtimes and preliminary estimates are labelled in context.",
      },
    ],
  },
  zh: {
    pageTitle: "应用实验室 —— 植物科学与 AI 的免费浏览器工具",
    pageDescription:
      "十四个免费浏览器工具，覆盖田间数据、作物研究、图像处理、可视化与 AI 辅助工作流。无需安装。",
    eyebrow: "数字植物表型平台",
    title: "面向田间数据、成像与 AI 的专注工作台。",
    subtitle:
      "十四个围绕植物科学工作流打造的实用浏览器工具。每个工作台共用同一套克制、响应式的界面，同时把各自的专业控件放在触手可及的位置。",
    statsLabel: "应用实验室概览",
    statTools: "个浏览器工具",
    statAreas: "个工作流领域",
    statByok: "自选 AI 模型",
    catalogEyebrow: "探索工具箱",
    catalogTitle: "选择一个工作流",
    catalogText:
      "按任务搜索，或按科研阶段筛选。每张卡片都会打开一个专用工作台。",
    searchLabel: "搜索工具",
    searchPlaceholder: "搜索工具或任务",
    clearSearch: "清除搜索",
    filterLabel: "按类别筛选工具",
    tool: "个工具",
    tools: "个工具",
    emptyTitle: "没有匹配的工具",
    emptyText: "请尝试更宽泛的关键词，或返回完整的应用实验室。",
    showAll: "显示全部工具",
    open: "打开",
    openTool: "打开工具",
    capabilities: "功能",
    trustLabel: "隐私与运行说明",
    trust: [
      {
        title: "尽可能本地优先",
        text: "除非工具明确指出使用了外部服务，否则文件与计算都留在浏览器内。",
      },
      {
        title: "你的模型，你的密钥",
        text: "AI 工具使用你在该次会话中自行配置的服务商与 API 凭据。",
      },
      {
        title: "标注科研边界",
        text: "科学假设、外部运行时与初步估算都会在上下文中标明。",
      },
    ],
  },
};

function AppCard({ app, isChinese, copy }) {
  const name = localizeApp(app.name, isChinese);
  const badges = localizeApp(app.badges, isChinese);
  return (
    <Link
      className={styles.appCard}
      to={app.route}
      data-tone={app.tone}
      aria-label={`${copy.open} ${name}`}
    >
      <div className={styles.cardTopline}>
        <span className={styles.appIcon} aria-hidden="true">
          {app.icon}
        </span>
        <span className={styles.categoryLabel}>
          {localizeApp(app.categoryLabel, isChinese)}
        </span>
      </div>

      <div className={styles.cardCopy}>
        <Heading as="h2" className={styles.cardTitle}>
          {localizeApp(app.shortName, isChinese)}
        </Heading>
        <p className={styles.cardDescription}>
          {localizeApp(app.description, isChinese)}
        </p>
      </div>

      <ul className={styles.cardBadges} aria-label={copy.capabilities}>
        {badges.slice(0, 3).map((badge) => (
          <li key={badge}>{badge}</li>
        ))}
      </ul>

      <div className={styles.cardAction} aria-hidden="true">
        <span>{copy.openTool}</span>
        <span className={styles.cardArrow}>↗</span>
      </div>
    </Link>
  );
}

export default function AppHub() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? COPY.zh : COPY.en;
  const [activeCategory, setActiveCategory] = useState("all");
  const [query, setQuery] = useState("");
  const filtersRef = useRef(null);
  const [sliderStyle, setSliderStyle] = useState({
    left: 0,
    width: 0,
    opacity: 0,
  });

  useEffect(() => {
    if (!filtersRef.current) return;
    const activeEl = filtersRef.current.querySelector(
      `.${styles.filterActive}`
    );
    if (activeEl) {
      setSliderStyle({
        left: activeEl.offsetLeft,
        width: activeEl.offsetWidth,
        opacity: 1,
      });
    } else {
      setSliderStyle((prev) => ({ ...prev, opacity: 0 }));
    }
  }, [activeCategory]);

  const visibleApps = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return APP_MANIFEST.filter((app) => {
      const matchesCategory =
        activeCategory === "all" || app.category === activeCategory;
      // Index both locales so a Chinese query still finds a tool on the
      // English site, and vice versa.
      const searchText = [
        app.name.en,
        app.name.zh,
        app.shortName.en,
        app.shortName.zh,
        app.description.en,
        app.description.zh,
        app.categoryLabel.en,
        app.categoryLabel.zh,
        ...app.badges.en,
        ...app.badges.zh,
        ...app.keywords,
      ]
        .join(" ")
        .toLowerCase();

      return (
        matchesCategory &&
        (!normalizedQuery || searchText.includes(normalizedQuery))
      );
    });
  }, [activeCategory, query]);

  const clearFilters = () => {
    setActiveCategory("all");
    setQuery("");
  };

  return (
    <Layout title={copy.pageTitle} description={copy.pageDescription}>
      <main className={styles.page}>
        <div className={styles.ambient} aria-hidden="true">
          <span className={styles.orbitOne} />
          <span className={styles.orbitTwo} />
          <span className={styles.gridGlow} />
        </div>

        <section className={styles.hero} aria-labelledby="app-lab-title">
          <div className={styles.heroCopy}>
            <p className={styles.eyebrow}>
              <span className={styles.liveDot} aria-hidden="true" />
              {copy.eyebrow}
            </p>
            <Heading as="h1" className={styles.title} id="app-lab-title">
              {copy.title}
            </Heading>
            <p className={styles.subtitle}>{copy.subtitle}</p>

            <dl className={styles.stats} aria-label={copy.statsLabel}>
              <div>
                <dt>{APP_MANIFEST.length}</dt>
                <dd>{copy.statTools}</dd>
              </div>
              <div>
                <dt>{APP_CATEGORIES.length - 1}</dt>
                <dd>{copy.statAreas}</dd>
              </div>
              <div>
                <dt>BYOK</dt>
                <dd>{copy.statByok}</dd>
              </div>
            </dl>
          </div>

          <div className={styles.heroVisual} aria-hidden="true">
            <div className={styles.visualCore}>
              <span>APP</span>
              <strong>LAB</strong>
            </div>
            <span className={styles.visualNode} data-node="field">
              GPS
            </span>
            <span className={styles.visualNode} data-node="vision">
              CV
            </span>
            <span className={styles.visualNode} data-node="ai">
              AI
            </span>
            <span className={styles.visualNode} data-node="data">
              CSV
            </span>
          </div>
        </section>

        <section className={styles.catalog} aria-labelledby="catalog-title">
          <div className={styles.catalogHeader}>
            <div>
              <p className={styles.sectionEyebrow}>{copy.catalogEyebrow}</p>
              <Heading as="h2" id="catalog-title">
                {copy.catalogTitle}
              </Heading>
              <p>{copy.catalogText}</p>
            </div>

            <label className={styles.searchBox}>
              <span className={styles.visuallyHidden}>{copy.searchLabel}</span>
              <span className={styles.searchIcon} aria-hidden="true" />
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={copy.searchPlaceholder}
                autoComplete="off"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label={copy.clearSearch}
                >
                  ×
                </button>
              )}
            </label>
          </div>

          <div className={styles.filterRow}>
            <div
              className={styles.filters}
              ref={filtersRef}
              aria-label={copy.filterLabel}
            >
              <div
                className={styles.glassSlider}
                style={{
                  left: `${sliderStyle.left}px`,
                  width: `${sliderStyle.width}px`,
                  opacity: sliderStyle.opacity,
                }}
              />
              {APP_CATEGORIES.map((category) => (
                <button
                  key={category.id}
                  type="button"
                  className={
                    activeCategory === category.id
                      ? styles.filterActive
                      : undefined
                  }
                  aria-pressed={activeCategory === category.id}
                  onClick={() => setActiveCategory(category.id)}
                >
                  {localizeApp(category.label, isChinese)}
                </button>
              ))}
            </div>
            <p className={styles.resultCount} aria-live="polite">
              {visibleApps.length}{" "}
              {visibleApps.length === 1 ? copy.tool : copy.tools}
            </p>
          </div>

          {visibleApps.length > 0 ? (
            <div className={styles.appGrid}>
              {visibleApps.map((app) => (
                <AppCard
                  key={app.id}
                  app={app}
                  isChinese={isChinese}
                  copy={copy}
                />
              ))}
            </div>
          ) : (
            <div className={styles.emptyState}>
              <span aria-hidden="true">⌕</span>
              <Heading as="h2">{copy.emptyTitle}</Heading>
              <p>{copy.emptyText}</p>
              <button type="button" onClick={clearFilters}>
                {copy.showAll}
              </button>
            </div>
          )}
        </section>

        <section className={styles.trustStrip} aria-label={copy.trustLabel}>
          {copy.trust.map((item) => (
            <div key={item.title}>
              <strong>{item.title}</strong>
              <span>{item.text}</span>
            </div>
          ))}
        </section>

        <div className={styles.citationWrap}>
          <CitationNotice />
        </div>
      </main>
    </Layout>
  );
}
