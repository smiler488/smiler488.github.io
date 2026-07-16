import React from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {
  navigatorCategories,
  navigatorLinks,
  navigatorUpdated,
} from "../../data/navigatorData";
import styles from "./styles.module.css";

const pageCopy = {
  en: {
    pageTitle: "Research Navigator",
    pageDescription:
      "A privacy-reviewed research web directory for AI, plant phenotyping, crop modeling, remote sensing, and scientific work.",
    eyebrow: "SMILER · RESEARCH NAVIGATOR",
    updated: `Curated ${navigatorUpdated.en}`,
    title: "Research Navigator",
    intro:
      "A compact, privacy-reviewed directory for AI for Science, crop modeling, plant phenotyping, remote sensing, and everyday research.",
    linkStat: "public links",
    categoryStat: "categories",
    policyStat: "privacy reviewed",
    browseTitle: "Browse categories",
    browseHint: "Jump directly to a section",
    categoryNavLabel: "Research navigator categories",
    resources: "sites",
    searchLabel: "Search the research navigator",
    searchPlaceholder: "Search sites, domains, or topics…",
    shortcut: "Press /",
    showing: "Showing",
    of: "of",
    links: "links",
    clear: "Clear search",
    noResultsTitle: "No destination matches that search.",
    noResultsBody:
      "Try a broader topic, another spelling, or return to the complete directory.",
    opensNewTab: "opens in a new tab",
    curationTitle: "Public by design.",
    curationBody:
      "The original browser export stays local. This page only contains manually reviewed public links; login pages, dashboards, personal identifiers, session parameters, direct IPs, and uncertain download sources are excluded.",
    privacyLink: "Read the privacy policy",
  },
  zh: {
    pageTitle: "科研网址导航",
    pageDescription:
      "面向 AI、植物表型、作物模型、遥感与科研工作的隐私审查型网址导航。",
    eyebrow: "SMILER · 科研网址导航",
    updated: `整理于 ${navigatorUpdated.zh}`,
    title: "科研网址导航",
    intro:
      "为 AI for Science、作物模型、植物表型、遥感与日常科研整理的紧凑型公开网址库，所有链接均经过隐私审查。",
    linkStat: "个公开网址",
    categoryStat: "个专业分类",
    policyStat: "已做隐私审查",
    browseTitle: "分类导航",
    browseHint: "点击直达对应分区",
    categoryNavLabel: "科研网址分类",
    resources: "个站点",
    searchLabel: "搜索科研网址导航",
    searchPlaceholder: "搜索网站、域名或主题…",
    shortcut: "按 /",
    showing: "当前显示",
    of: "/",
    links: "个网址",
    clear: "清除搜索",
    noResultsTitle: "没有找到匹配的网址。",
    noResultsBody: "请尝试更宽泛的主题、不同写法，或返回完整导航。",
    opensNewTab: "将在新标签页打开",
    curationTitle: "从一开始就保护隐私。",
    curationBody:
      "原始浏览器导出文件始终保留在本地。本页只收录人工审查后的公开网址；登录页、控制台、个人标识、会话参数、直接 IP 和来源不明的下载入口均已排除。",
    privacyLink: "查看隐私政策",
  },
};

function localize(value, isChinese) {
  return isChinese ? value.zh : value.en;
}

function getDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function getSiteMark(domain) {
  const root = domain.split(".")[0].replace(/[^a-z0-9]/gi, "");
  return root.slice(0, 2).toUpperCase() || "↗";
}

function CategoryIcon({ categoryId }) {
  const commonProps = {
    width: 24,
    height: 24,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
  };

  switch (categoryId) {
    case "ai-ecosystem":
      return (
        <svg {...commonProps}>
          <path d="M12 3.2 13.8 8l4.8 1.8-4.8 1.8-1.8 4.8-1.8-4.8-4.8-1.8L10.2 8 12 3.2Z" />
          <path d="m18.2 15 .8 2.1 2.1.8-2.1.8-.8 2.1-.8-2.1-2.1-.8 2.1-.8.8-2.1Z" />
        </svg>
      );
    case "ai-science":
      return (
        <svg {...commonProps}>
          <circle cx="12" cy="12" r="1.8" />
          <ellipse cx="12" cy="12" rx="9" ry="3.7" />
          <ellipse
            cx="12"
            cy="12"
            rx="3.7"
            ry="9"
            transform="rotate(40 12 12)"
          />
          <ellipse
            cx="12"
            cy="12"
            rx="3.7"
            ry="9"
            transform="rotate(-40 12 12)"
          />
        </svg>
      );
    case "digital-crops":
      return (
        <svg {...commonProps}>
          <path d="M12 21V9" />
          <path d="M12 13c-4.7 0-7.5-2.3-7.5-6.5 4.7 0 7.5 2.3 7.5 6.5Z" />
          <path d="M12 9c4.7 0 7.5-2.3 7.5-6.5C14.8 2.5 12 4.8 12 9Z" />
          <path d="M8.5 21h7" />
        </svg>
      );
    case "phenotyping":
      return (
        <svg {...commonProps}>
          <path d="M8 3H4a1 1 0 0 0-1 1v4M16 3h4a1 1 0 0 1 1 1v4M8 21H4a1 1 0 0 1-1-1v-4M16 21h4a1 1 0 0 0 1-1v-4" />
          <path d="M12 17V9" />
          <path d="M12 12c-3.2 0-5-1.6-5-4.5 3.2 0 5 1.6 5 4.5ZM12 9c3.2 0 5-1.6 5-4.5-3.2 0-5 1.6-5 4.5Z" />
        </svg>
      );
    case "geo-remote":
      return (
        <svg {...commonProps}>
          <circle cx="12" cy="12" r="9" />
          <path d="M3.5 9h17M3.5 15h17M12 3c2.4 2.5 3.5 5.5 3.5 9S14.4 18.5 12 21c-2.4-2.5-3.5-5.5-3.5-9S9.6 5.5 12 3Z" />
        </svg>
      );
    case "data-orgs":
      return (
        <svg {...commonProps}>
          <ellipse cx="12" cy="5" rx="7.5" ry="3" />
          <path d="M4.5 5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V5M4.5 11v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6" />
        </svg>
      );
    case "physiology":
      return (
        <svg {...commonProps}>
          <path d="M3 12h4l2.2-5 4.2 10 2.2-5H21" />
          <path d="M18.5 5.5c-3.5.1-5.3 1.8-5.3 5.1 3.5-.1 5.3-1.8 5.3-5.1Z" />
        </svg>
      );
    case "writing-viz":
      return (
        <svg {...commonProps}>
          <path d="M4 20h16M6 17V9M11 17V4M16 17v-6M21 17V7" />
          <path d="m3.5 5.5 3-3 2 2-3 3-2.5.5.5-2.5Z" />
        </svg>
      );
    default:
      return (
        <svg {...commonProps}>
          <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H11v16H6.5A2.5 2.5 0 0 0 4 21.5v-16ZM20 5.5A2.5 2.5 0 0 0 17.5 3H13v16h4.5a2.5 2.5 0 0 1 2.5 2.5v-16Z" />
        </svg>
      );
  }
}

function LinkCard({ item, category, copy }) {
  const domain = getDomain(item.url);

  return (
    <li className={`${styles.linkItem} ${styles[category.id]}`}>
      <Link
        className={`${styles.linkCard} no-external-icon`}
        to={item.url}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`${item.title} — ${copy.opensNewTab}`}
      >
        <span className={styles.siteMark} aria-hidden="true">
          {getSiteMark(domain)}
        </span>
        <span className={styles.linkIdentity}>
          <span className={styles.linkTitle}>{item.title}</span>
          <span className={styles.linkDomain}>{domain}</span>
        </span>
        <span className={styles.linkArrow} aria-hidden="true">
          ↗
        </span>
      </Link>
    </li>
  );
}

function ShieldIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

export default function NavigatorPage() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? pageCopy.zh : pageCopy.en;
  const [query, setQuery] = React.useState("");
  const [activeCategory, setActiveCategory] = React.useState(
    navigatorCategories[0].id
  );
  const searchRef = React.useRef(null);

  const categoryCounts = React.useMemo(
    () =>
      navigatorLinks.reduce((counts, item) => {
        counts[item.category] = (counts[item.category] || 0) + 1;
        return counts;
      }, {}),
    []
  );

  const filteredLinks = React.useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    if (!normalizedQuery) return navigatorLinks;

    return navigatorLinks.filter((item) => {
      const categoryItem = navigatorCategories.find(
        (entry) => entry.id === item.category
      );
      const searchText = [
        item.title,
        getDomain(item.url),
        localize(categoryItem.label, isChinese),
        localize(categoryItem.description, isChinese),
        ...categoryItem.keywords,
      ]
        .join(" ")
        .toLocaleLowerCase();

      return searchText.includes(normalizedQuery);
    });
  }, [isChinese, query]);

  const visibleGroups = React.useMemo(
    () =>
      navigatorCategories
        .map((categoryItem) => ({
          category: categoryItem,
          items: filteredLinks.filter(
            (item) => item.category === categoryItem.id
          ),
        }))
        .filter((group) => group.items.length > 0),
    [filteredLinks]
  );

  React.useEffect(() => {
    function handleKeyDown(event) {
      const target = event.target;
      const isEditing =
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT");

      if (event.key === "/" && !isEditing && !event.metaKey && !event.ctrlKey) {
        event.preventDefault();
        searchRef.current?.focus();
      }

      if (
        event.key === "Escape" &&
        document.activeElement === searchRef.current
      ) {
        setQuery("");
        searchRef.current?.blur();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  React.useEffect(() => {
    if (query.trim() || typeof IntersectionObserver === "undefined") return;

    const sections = Array.from(
      document.querySelectorAll("[data-navigator-category]")
    );
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort(
            (first, second) =>
              Math.abs(first.boundingClientRect.top) -
              Math.abs(second.boundingClientRect.top)
          );

        if (visible[0]?.target.dataset.navigatorCategory) {
          setActiveCategory(visible[0].target.dataset.navigatorCategory);
        }
      },
      { rootMargin: "-14% 0px -72% 0px", threshold: 0.01 }
    );

    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, [query]);

  function scrollToCategory(categoryId) {
    setQuery("");
    setActiveCategory(categoryId);

    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        const prefersReducedMotion = window.matchMedia(
          "(prefers-reduced-motion: reduce)"
        ).matches;
        document.getElementById(`navigator-${categoryId}`)?.scrollIntoView({
          behavior: prefersReducedMotion ? "auto" : "smooth",
          block: "start",
        });
      });
    });
  }

  function clearSearch() {
    setQuery("");
    window.requestAnimationFrame(() => searchRef.current?.focus());
  }

  return (
    <Layout title={copy.pageTitle} description={copy.pageDescription}>
      <main className={styles.page}>
        <div className={styles.shell}>
          <header className={styles.hero}>
            <div className={styles.heroTop}>
              <div className={styles.heroCopy}>
                <div className={styles.eyebrowRow}>
                  <span className={styles.eyebrow}>{copy.eyebrow}</span>
                  <span className={styles.updatedPill}>{copy.updated}</span>
                </div>
                <Heading as="h1" className={styles.heroTitle}>
                  {copy.title}
                </Heading>
                <p className={styles.heroDescription}>{copy.intro}</p>
              </div>

              <dl className={styles.heroStats} aria-label={copy.pageTitle}>
                <div>
                  <dd>{navigatorLinks.length}</dd>
                  <dt>{copy.linkStat}</dt>
                </div>
                <div>
                  <dd>{navigatorCategories.length}</dd>
                  <dt>{copy.categoryStat}</dt>
                </div>
                <div className={styles.policyStat}>
                  <dd aria-hidden="true">✓</dd>
                  <dt>{copy.policyStat}</dt>
                </div>
              </dl>
            </div>

            <div className={styles.searchRow} role="search">
              <label className={styles.searchField}>
                <span className={styles.srOnly}>{copy.searchLabel}</span>
                <span className={styles.searchBox}>
                  <svg
                    width="19"
                    height="19"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    aria-hidden="true"
                  >
                    <circle cx="11" cy="11" r="7" />
                    <path d="m20 20-4-4" />
                  </svg>
                  <input
                    ref={searchRef}
                    type="search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder={copy.searchPlaceholder}
                    autoComplete="off"
                  />
                  {query ? (
                    <button
                      type="button"
                      className={styles.searchClear}
                      onClick={clearSearch}
                      aria-label={copy.clear}
                    >
                      ×
                    </button>
                  ) : (
                    <kbd>{copy.shortcut}</kbd>
                  )}
                </span>
              </label>

              <div className={styles.resultBar}>
                <p aria-live="polite" aria-atomic="true">
                  {copy.showing} <strong>{filteredLinks.length}</strong>{" "}
                  {copy.of} {navigatorLinks.length} {copy.links}
                </p>
                {query ? (
                  <button type="button" onClick={clearSearch}>
                    {copy.clear}
                  </button>
                ) : null}
              </div>
            </div>
          </header>

          <div className={styles.directoryLayout}>
            <aside className={styles.categoryRail}>
              <div className={styles.railHeading}>
                <strong>{copy.browseTitle}</strong>
                <span>{copy.browseHint}</span>
              </div>
              <nav
                className={styles.categoryList}
                aria-label={copy.categoryNavLabel}
              >
                {navigatorCategories.map((item) => {
                  const isActive = activeCategory === item.id && !query;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      className={`${styles.categoryButton} ${styles[item.id]} ${
                        isActive ? styles.categoryButtonActive : ""
                      }`}
                      onClick={() => scrollToCategory(item.id)}
                      aria-pressed={isActive}
                      aria-controls={`navigator-${item.id}`}
                    >
                      <span className={styles.categoryIcon} aria-hidden="true">
                        <CategoryIcon categoryId={item.id} />
                      </span>
                      <span className={styles.categoryText}>
                        <strong>{localize(item.label, isChinese)}</strong>
                        <small>
                          {categoryCounts[item.id]} {copy.resources}
                        </small>
                      </span>
                      <span
                        className={styles.categoryChevron}
                        aria-hidden="true"
                      >
                        ›
                      </span>
                    </button>
                  );
                })}
              </nav>
              <div className={styles.railPrivacy}>
                <ShieldIcon />
                <Link to="/privacy">{copy.privacyLink} →</Link>
              </div>
            </aside>

            <section
              className={styles.directoryPanel}
              aria-label={copy.pageTitle}
            >
              {visibleGroups.length > 0 ? (
                <div className={styles.groupList}>
                  {visibleGroups.map((group) => (
                    <section
                      key={group.category.id}
                      id={`navigator-${group.category.id}`}
                      data-navigator-category={group.category.id}
                      className={`${styles.linkGroup} ${
                        styles[group.category.id]
                      }`}
                      aria-labelledby={`navigator-${group.category.id}-title`}
                    >
                      <div className={styles.groupHeader}>
                        <span className={styles.groupIcon} aria-hidden="true">
                          <CategoryIcon categoryId={group.category.id} />
                        </span>
                        <div className={styles.groupCopy}>
                          <Heading
                            as="h2"
                            id={`navigator-${group.category.id}-title`}
                            className={styles.groupTitle}
                          >
                            {localize(group.category.label, isChinese)}
                          </Heading>
                          <p>
                            {localize(group.category.description, isChinese)}
                          </p>
                        </div>
                        <span
                          className={styles.groupCount}
                          aria-label={`${group.items.length} ${copy.resources}`}
                        >
                          {group.items.length}
                        </span>
                      </div>
                      <ul className={styles.linkGrid}>
                        {group.items.map((item) => (
                          <LinkCard
                            key={item.url}
                            item={item}
                            category={group.category}
                            copy={copy}
                          />
                        ))}
                      </ul>
                    </section>
                  ))}
                </div>
              ) : (
                <div className={styles.emptyState}>
                  <span aria-hidden="true">⌕</span>
                  <Heading as="h2">{copy.noResultsTitle}</Heading>
                  <p>{copy.noResultsBody}</p>
                  <button type="button" onClick={clearSearch}>
                    {copy.clear}
                  </button>
                </div>
              )}

              <aside className={styles.curationNote}>
                <span className={styles.noteIcon} aria-hidden="true">
                  <ShieldIcon />
                </span>
                <p>
                  <strong>{copy.curationTitle}</strong> {copy.curationBody}{" "}
                  <Link to="/privacy">{copy.privacyLink} →</Link>
                </p>
              </aside>
            </section>
          </div>
        </div>
      </main>
    </Layout>
  );
}
