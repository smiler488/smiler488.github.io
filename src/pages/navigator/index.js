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
    eyebrow: "SMILER RESEARCH NAVIGATOR",
    updated: `Curated ${navigatorUpdated.en}`,
    title: "A calmer path through the research web.",
    intro:
      "A focused, public selection from a working research library—cleaned for duplicates, private accounts, and unsafe parameters, then organized around AI for Science and digital agriculture.",
    linkStat: "public links",
    laneStat: "research lanes",
    policyStat: "privacy first",
    lanesEyebrow: "Browse by workflow",
    lanesTitle: "Nine routes from an idea to evidence.",
    lanesDescription:
      "Start with a research lane or search across the complete collection. Each destination opens directly at its source.",
    resources: "resources",
    catalogEyebrow: "Link library",
    catalogTitle: "Find the next useful destination.",
    catalogDescription:
      "Search by site, domain, topic, or discipline. Press / anywhere on this page to focus search.",
    searchLabel: "Search the research navigator",
    searchPlaceholder: "Search sites, domains, or topics…",
    shortcut: "Press /",
    all: "All",
    showing: "Showing",
    of: "of",
    links: "links",
    clear: "Clear filters",
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
    eyebrow: "SMILER 科研网址导航",
    updated: `整理于 ${navigatorUpdated.zh}`,
    title: "让科研网络，变得更清晰。",
    intro:
      "从真实科研收藏夹中筛选公开资源，去除重复项、私人账户与不安全参数，并围绕 AI for Science 和数字农业重新组织。",
    linkStat: "个公开入口",
    laneStat: "条科研路线",
    policyStat: "隐私优先",
    lanesEyebrow: "按工作流浏览",
    lanesTitle: "从研究想法到可靠证据的九条路线。",
    lanesDescription:
      "选择一个科研方向，或在完整收藏中搜索；每个入口都直接指向资源来源。",
    resources: "个资源",
    catalogEyebrow: "网址库",
    catalogTitle: "找到下一站有用的资源。",
    catalogDescription:
      "可按站点、域名、主题或学科搜索；在本页任意位置按 / 即可聚焦搜索框。",
    searchLabel: "搜索科研网址导航",
    searchPlaceholder: "搜索网站、域名或主题…",
    shortcut: "按 /",
    all: "全部",
    showing: "当前显示",
    of: "/",
    links: "个网址",
    clear: "清除筛选",
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
        <Heading as="h3" className={styles.linkTitle}>
          {item.title}
        </Heading>
        <span className={styles.linkDomain}>{domain}</span>
      </span>
      <span className={styles.linkArrow} aria-hidden="true">
        ↗
      </span>
      <span
        className={`${styles.cardAccent} ${styles[category.id]}`}
        aria-hidden="true"
      />
    </Link>
  );
}

export default function NavigatorPage() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? pageCopy.zh : pageCopy.en;
  const [query, setQuery] = React.useState("");
  const [category, setCategory] = React.useState("all");
  const searchRef = React.useRef(null);
  const catalogRef = React.useRef(null);

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

    return navigatorLinks.filter((item) => {
      if (category !== "all" && item.category !== category) return false;
      if (!normalizedQuery) return true;

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
  }, [category, isChinese, query]);

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

  function selectLane(categoryId) {
    setCategory(categoryId);
    window.requestAnimationFrame(() => {
      catalogRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  }

  function clearFilters() {
    setQuery("");
    setCategory("all");
  }

  const hasFilters = query.length > 0 || category !== "all";

  return (
    <Layout title={copy.pageTitle} description={copy.pageDescription}>
      <main className={styles.page}>
        <div className={styles.ambientOne} aria-hidden="true" />
        <div className={styles.ambientTwo} aria-hidden="true" />

        <div className={styles.shell}>
          <header className={styles.hero}>
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
                <dt>{copy.linkStat}</dt>
                <dd>{navigatorLinks.length}</dd>
              </div>
              <div>
                <dt>{copy.laneStat}</dt>
                <dd>{navigatorCategories.length}</dd>
              </div>
              <div className={styles.policyStat}>
                <dt>{copy.policyStat}</dt>
                <dd aria-hidden="true">✓</dd>
              </div>
            </dl>
          </header>

          <section
            className={styles.lanesSection}
            aria-labelledby="navigator-lanes-title"
          >
            <div className={styles.sectionHeading}>
              <div>
                <span className={styles.sectionEyebrow}>
                  {copy.lanesEyebrow}
                </span>
                <Heading
                  as="h2"
                  id="navigator-lanes-title"
                  className={styles.sectionTitle}
                >
                  {copy.lanesTitle}
                </Heading>
              </div>
              <p>{copy.lanesDescription}</p>
            </div>

            <div className={styles.laneGrid}>
              {navigatorCategories.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`${styles.laneCard} ${styles[item.id]}`}
                  onClick={() => selectLane(item.id)}
                  aria-pressed={category === item.id}
                  aria-controls="navigator-catalog"
                >
                  <span className={styles.laneIcon}>
                    <CategoryIcon categoryId={item.id} />
                  </span>
                  <span className={styles.laneCopy}>
                    <span className={styles.laneTitle}>
                      {localize(item.label, isChinese)}
                    </span>
                    <span className={styles.laneDescription}>
                      {localize(item.description, isChinese)}
                    </span>
                  </span>
                  <span className={styles.laneMeta}>
                    {categoryCounts[item.id]} {copy.resources}
                    <span aria-hidden="true">→</span>
                  </span>
                </button>
              ))}
            </div>
          </section>

          <section
            ref={catalogRef}
            className={styles.catalogSection}
            id="navigator-catalog"
            aria-labelledby="navigator-catalog-title"
          >
            <div className={styles.sectionHeading}>
              <div>
                <span className={styles.sectionEyebrow}>
                  {copy.catalogEyebrow}
                </span>
                <Heading
                  as="h2"
                  id="navigator-catalog-title"
                  className={styles.sectionTitle}
                >
                  {copy.catalogTitle}
                </Heading>
              </div>
              <p>{copy.catalogDescription}</p>
            </div>

            <div className={styles.filterPanel}>
              <label className={styles.searchField}>
                <span className={styles.searchLabel}>{copy.searchLabel}</span>
                <span className={styles.searchBox}>
                  <svg
                    width="20"
                    height="20"
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
                      onClick={() => setQuery("")}
                      aria-label={copy.clear}
                    >
                      ×
                    </button>
                  ) : (
                    <kbd>{copy.shortcut}</kbd>
                  )}
                </span>
              </label>

              <div
                className={styles.categoryScroller}
                aria-label={copy.lanesEyebrow}
              >
                <button
                  type="button"
                  className={
                    category === "all"
                      ? styles.filterChipActive
                      : styles.filterChip
                  }
                  onClick={() => setCategory("all")}
                  aria-pressed={category === "all"}
                >
                  {copy.all}
                  <span>{navigatorLinks.length}</span>
                </button>
                {navigatorCategories.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={
                      category === item.id
                        ? styles.filterChipActive
                        : styles.filterChip
                    }
                    onClick={() => setCategory(item.id)}
                    aria-pressed={category === item.id}
                  >
                    {localize(item.label, isChinese)}
                    <span>{categoryCounts[item.id]}</span>
                  </button>
                ))}
              </div>

              <div className={styles.resultBar}>
                <p aria-live="polite" aria-atomic="true">
                  {copy.showing} <strong>{filteredLinks.length}</strong>{" "}
                  {copy.of} {navigatorLinks.length} {copy.links}
                </p>
                {hasFilters ? (
                  <button type="button" onClick={clearFilters}>
                    {copy.clear}
                  </button>
                ) : null}
              </div>
            </div>

            {visibleGroups.length > 0 ? (
              <div className={styles.groupList}>
                {visibleGroups.map((group) => (
                  <section
                    key={group.category.id}
                    className={`${styles.linkGroup} ${
                      styles[group.category.id]
                    }`}
                    aria-labelledby={`navigator-${group.category.id}`}
                  >
                    <div className={styles.groupHeader}>
                      <span className={styles.groupIcon}>
                        <CategoryIcon categoryId={group.category.id} />
                      </span>
                      <div>
                        <Heading
                          as="h2"
                          id={`navigator-${group.category.id}`}
                          className={styles.groupTitle}
                        >
                          {localize(group.category.label, isChinese)}
                        </Heading>
                        <p>{localize(group.category.description, isChinese)}</p>
                      </div>
                      <span className={styles.groupCount}>
                        {group.items.length}
                      </span>
                    </div>
                    <div className={styles.linkGrid}>
                      {group.items.map((item) => (
                        <LinkCard
                          key={item.url}
                          item={item}
                          category={group.category}
                          copy={copy}
                        />
                      ))}
                    </div>
                  </section>
                ))}
              </div>
            ) : (
              <div className={styles.emptyState}>
                <span aria-hidden="true">⌕</span>
                <Heading as="h2">{copy.noResultsTitle}</Heading>
                <p>{copy.noResultsBody}</p>
                <button type="button" onClick={clearFilters}>
                  {copy.clear}
                </button>
              </div>
            )}
          </section>

          <aside className={styles.curationNote}>
            <span className={styles.noteIcon} aria-hidden="true">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
                <path d="m9 12 2 2 4-4" />
              </svg>
            </span>
            <div>
              <Heading as="h2">{copy.curationTitle}</Heading>
              <p>{copy.curationBody}</p>
              <Link to="/privacy">{copy.privacyLink} →</Link>
            </div>
          </aside>
        </div>
      </main>
    </Layout>
  );
}
