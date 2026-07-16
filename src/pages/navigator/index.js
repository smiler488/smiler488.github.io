import React from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import {
  navigatorCategories,
  navigatorGroups,
  navigatorLinks,
  navigatorUpdated,
} from "../../data/navigatorData";
import styles from "./styles.module.css";

const pageCopy = {
  en: {
    pageTitle: "Workspace Navigator",
    pageDescription:
      "A personal workspace of everyday entry points: campus and postdoc systems, literature and journals, research data, and downtime.",
    eyebrow: "SMILER · WORKSPACE",
    updated: `Curated ${navigatorUpdated.en}`,
    title: "Workspace Navigator",
    intro:
      "My daily desk in one page: university and postdoc systems for work, literature, journals and data platforms for research, and a short list of places to unwind.",
    linkStat: "public links",
    categoryStat: "categories",
    policyStat: "privacy reviewed",
    browseTitle: "Browse the desk",
    browseHint: "Work · Research · Daily",
    categoryNavLabel: "Workspace categories",
    resources: "sites",
    searchLabel: "Search the workspace navigator",
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
    curationTitle: "Public entry points only.",
    curationBody:
      "The original browser export stays local. This page lists only the public front doors of the systems I use—sign-in still happens on their side. Personal identifiers, session parameters, direct IPs, and uncertain download sources are excluded.",
    privacyLink: "Read the privacy policy",
    resourcesLink: "Looking for study material? Visit Resources",
  },
  zh: {
    pageTitle: "个人工作台导航",
    pageDescription:
      "个人日常入口工作台：校内与博士后系统、文献与期刊、科研数据平台，以及休闲站点。",
    eyebrow: "SMILER · 个人工作台",
    updated: `整理于 ${navigatorUpdated.zh}`,
    title: "个人工作台导航",
    intro:
      "把每天要用的入口收在一页：工作用的学校与博士后系统，科研用的文献、期刊与数据平台，以及少量放松用的站点。",
    linkStat: "个公开网址",
    categoryStat: "个分类",
    policyStat: "已做隐私审查",
    browseTitle: "工作台分区",
    browseHint: "工作 · 学习科研 · 日常",
    categoryNavLabel: "工作台分类",
    resources: "个站点",
    searchLabel: "搜索个人工作台",
    searchPlaceholder: "搜索网站、域名或主题…",
    shortcut: "按 /",
    showing: "当前显示",
    of: "/",
    links: "个网址",
    clear: "清除搜索",
    noResultsTitle: "没有找到匹配的网址。",
    noResultsBody: "请尝试更宽泛的主题、不同写法，或返回完整导航。",
    opensNewTab: "将在新标签页打开",
    curationTitle: "只收录公开入口。",
    curationBody:
      "原始浏览器导出文件始终保留在本地。本页只列出这些系统的公开入口，登录仍在对方站点完成；个人标识、会话参数、直接 IP 和来源不明的下载入口均已排除。",
    privacyLink: "查看隐私政策",
    resourcesLink: "想找学习资料？前往资源页",
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
    case "shzu":
      return (
        <svg {...commonProps}>
          <path d="M3 10.5 12 5l9 5.5" />
          <path d="M5 10.5V20h14v-9.5" />
          <path d="M9.5 20v-5.5h5V20" />
        </svg>
      );
    case "cau-postdoc":
      return (
        <svg {...commonProps}>
          <path d="M12 4 2.8 8.4 12 12.8l9.2-4.4L12 4Z" />
          <path d="M6.5 10.6v4.6c0 1.6 2.6 2.9 5.5 2.9s5.5-1.3 5.5-2.9v-4.6" />
          <path d="M21.2 8.4V14" />
        </svg>
      );
    case "funding-policy":
      return (
        <svg {...commonProps}>
          <path d="M4 20h16" />
          <path d="M5.5 20V10m4.5 10V10m4 10V10m4.5 10V10" />
          <path d="M12 3.2 21 8H3l9-4.8Z" />
        </svg>
      );
    case "literature":
      return (
        <svg {...commonProps}>
          <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H11v14H6.5A2.5 2.5 0 0 0 4 19.5v-14Z" />
          <path d="M20 5.5A2.5 2.5 0 0 0 17.5 3H13v14h4.5a2.5 2.5 0 0 1 2.5 2.5v-14Z" />
          <circle cx="17.5" cy="16.5" r="3.5" />
          <path d="m20.2 19.2 1.8 1.8" />
        </svg>
      );
    case "journals":
      return (
        <svg {...commonProps}>
          <path d="M5 3.8h11.5a2 2 0 0 1 2 2v14.4H7a2 2 0 0 1-2-2V3.8Z" />
          <path d="M8.5 8h7M8.5 11.5h7M8.5 15h4" />
          <path d="M18.5 8H21v12.2H7" />
        </svg>
      );
    case "data-platforms":
      return (
        <svg {...commonProps}>
          <ellipse cx="12" cy="5" rx="7.5" ry="3" />
          <path d="M4.5 5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V5M4.5 11v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6" />
        </svg>
      );
    case "toolbox":
      return (
        <svg {...commonProps}>
          <path d="M3 9h18v10.5a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 19.5V9Z" />
          <path d="M8.5 9V5.5A1.5 1.5 0 0 1 10 4h4a1.5 1.5 0 0 1 1.5 1.5V9" />
          <path d="M3 13.5h18" />
        </svg>
      );
    case "build-ship":
      return (
        <svg {...commonProps}>
          <path d="m8.5 8.5-4 3.5 4 3.5" />
          <path d="m15.5 8.5 4 3.5-4 3.5" />
          <path d="m13.5 5-3 14" />
        </svg>
      );
    case "life":
      return (
        <svg {...commonProps}>
          <rect x="2.8" y="5" width="18.4" height="14" rx="2.2" />
          <path d="M2.8 9h18.4" />
          <path d="m10.5 12.2 4 2.3-4 2.3v-4.6Z" />
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

function LinkCard({ item, category, copy, isChinese }) {
  const title = localize(item.title, isChinese);
  const domain = getDomain(item.url);

  return (
    <li className={`${styles.linkItem} ${styles[category.id]}`}>
      <Link
        className={`${styles.linkCard} no-external-icon`}
        to={item.url}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`${title} — ${copy.opensNewTab}`}
      >
        <span className={styles.siteMark} aria-hidden="true">
          {getSiteMark(domain)}
        </span>
        <span className={styles.linkIdentity}>
          <span className={styles.linkTitle}>{title}</span>
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
        item.title.en,
        item.title.zh,
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
                {navigatorGroups.map((group) => (
                  <div key={group.id} className={styles.categoryGroup}>
                    <p className={styles.categoryGroupLabel}>
                      <strong>{localize(group.label, isChinese)}</strong>
                      <small>{localize(group.hint, isChinese)}</small>
                    </p>
                    {navigatorCategories
                      .filter((item) => item.group === group.id)
                      .map((item) => {
                        const isActive = activeCategory === item.id && !query;
                        return (
                          <button
                            key={item.id}
                            type="button"
                            className={`${styles.categoryButton} ${
                              styles[item.id]
                            } ${isActive ? styles.categoryButtonActive : ""}`}
                            onClick={() => scrollToCategory(item.id)}
                            aria-pressed={isActive}
                            aria-controls={`navigator-${item.id}`}
                          >
                            <span
                              className={styles.categoryIcon}
                              aria-hidden="true"
                            >
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
                  </div>
                ))}
              </nav>
              <div className={styles.railPrivacy}>
                <ShieldIcon />
                <Link to="/privacy">{copy.privacyLink} →</Link>
              </div>
              <div className={styles.railCrossLink}>
                <Link to="/resources">{copy.resourcesLink} →</Link>
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
                            isChinese={isChinese}
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
