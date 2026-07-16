import React from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import clsx from "clsx";
import styles from "./styles.module.css";
import {
  learningPaths,
  learningResources,
  resourceCategories,
  resourceGroups,
  resourceKinds,
  resourceLevels,
} from "../../data/resourcesData";

const levelOrder = ["beginner", "intermediate", "advanced"];

const pageCopy = {
  en: {
    pageTitle: "AI for Science & Agriculture Resources",
    pageDescription:
      "A curated learning hub for scientific machine learning, AI agents & LLMs, crop models, plant phenotyping, GeoAI, and 3D vision.",
    eyebrow: "Research learning hub",
    updated: "Curated and verified · July 2026",
    heroTitle: "Learn AI for Science & digital agriculture.",
    heroDescription:
      "A focused map of official courses, models, datasets, and tools for scientific machine learning, AI agents & LLMs, crop simulation, plant phenotyping, remote sensing, and 3D reconstruction.",
    statResources: "curated resources",
    statTopics: "focused topics",
    statSources: "official-first",
    statSourcesLabel: "source policy",
    statVerified: "July 2026",
    statVerifiedLabel: "last verified",
    pathsEyebrow: "Start with a goal",
    pathsTitle: "Three practical learning paths",
    pathsDescription:
      "Each path moves from a reliable starting point to a reproducible research workflow.",
    openPath: "Explore this path",
    catalogEyebrow: "Curated catalog",
    catalogTitle: "Find the right resource",
    catalogDescription:
      "Pick a dimension on the left—foundations, domain, or publishing—then read each topic from beginner to advanced.",
    browseTitle: "Browse by dimension",
    browseHint: "Base · Domain · Publish",
    categoryNavLabel: "Resource topics",
    railCount: "resources",
    navigatorLink: "My own daily workspace links",
    searchLabel: "Search resources",
    searchPlaceholder: "Search models, datasets, courses, or topics…",
    levelLabel: "Experience level",
    allLevels: "All levels",
    allTopics: "All topics",
    clear: "Clear filters",
    showing: "Showing",
    of: "of",
    resources: "resources",
    official: "Official",
    community: "Community",
    openSource: "Open source",
    updatedBadge: "Updated",
    openResource: "Open resource",
    newTab: "opens in a new tab",
    noResultsTitle: "No resources match those filters.",
    noResultsDescription:
      "Try a broader keyword, switch topics, or reset the experience level.",
    methodologyEyebrow: "How this list is maintained",
    methodologyTitle: "Useful, reproducible, and current—not merely popular.",
    methodologyDescription:
      "Official and maintainer-owned sources are prioritized. Resources are selected for learning value, reproducibility, and recent activity. Access terms can change, and research-model outputs still require domain validation before scientific or operational use.",
    methodologyPointOne:
      "Low-signal tool directories and stale links were removed.",
    methodologyPointTwo:
      "Licensing or account requirements are summarized on each card.",
    methodologyPointThree:
      "The catalog is designed to be reviewed and refreshed over time.",
  },
  zh: {
    pageTitle: "AI for Science 与农业学习资源",
    pageDescription:
      "面向科学机器学习、智能体与大模型、作物模型、植物表型、GeoAI 与三维视觉的精选学习资源中心。",
    eyebrow: "科研学习资源中心",
    updated: "精选并核验 · 2026 年 7 月",
    heroTitle: "学习 AI for Science 与数字农业。",
    heroDescription:
      "围绕科学机器学习、智能体与大模型、作物模拟、植物表型、遥感与三维重建，精选官方课程、模型、数据集和工具。",
    statResources: "项精选资源",
    statTopics: "个聚焦主题",
    statSources: "官方优先",
    statSourcesLabel: "来源原则",
    statVerified: "2026 年 7 月",
    statVerifiedLabel: "最近核验",
    pathsEyebrow: "从目标出发",
    pathsTitle: "三条可执行的学习路径",
    pathsDescription: "每条路径都从可靠起点出发，逐步进入可复现的科研工作流。",
    openPath: "查看这条路径",
    catalogEyebrow: "精选目录",
    catalogTitle: "找到合适的学习资源",
    catalogDescription:
      "先在左侧选择维度——基础、领域或表达发表，再在每个主题内由入门读到高级。",
    browseTitle: "按维度浏览",
    browseHint: "基础 · 领域 · 表达",
    categoryNavLabel: "资源主题",
    railCount: "项资源",
    navigatorLink: "查看我的日常工作台导航",
    searchLabel: "搜索资源",
    searchPlaceholder: "搜索模型、数据集、课程或主题…",
    levelLabel: "经验水平",
    allLevels: "全部难度",
    allTopics: "全部主题",
    clear: "清除筛选",
    showing: "显示",
    of: "/",
    resources: "项资源",
    official: "官方",
    community: "社区",
    openSource: "开源",
    updatedBadge: "更新于",
    openResource: "打开资源",
    newTab: "将在新标签页打开",
    noResultsTitle: "没有匹配当前条件的资源。",
    noResultsDescription: "请尝试更宽泛的关键词、切换主题或重置难度。",
    methodologyEyebrow: "维护原则",
    methodologyTitle: "重视实用、可复现与时效，而不是简单罗列热门工具。",
    methodologyDescription:
      "优先采用官方或维护团队的一手入口，并根据学习价值、可复现性和近期活跃度筛选。访问条款可能变化，科研模型的输出在用于科学结论或生产决策前仍需领域验证。",
    methodologyPointOne: "已移除低信息量的工具堆叠和失效链接。",
    methodologyPointTwo: "每张卡片概括许可、账户或访问要求。",
    methodologyPointThree: "目录结构便于后续持续核验和更新。",
  },
};

function localize(value, isChinese) {
  if (typeof value === "string") return value;
  return isChinese ? value.zh : value.en;
}

function ResourceCard({ resource, isChinese, copy }) {
  const category = resourceCategories.find(
    (item) => item.id === resource.category
  );
  const description = localize(resource.description, isChinese);
  const kind = localize(resourceKinds[resource.kind], isChinese);
  const level = localize(resourceLevels[resource.level], isChinese);
  const categoryLabel = localize(category.short, isChinese);
  const sourceLabel =
    resource.source === "community" ? copy.community : copy.official;

  return (
    <article className={clsx(styles.resourceCard, styles[resource.category])}>
      <div className={styles.cardTopline}>
        <div className={styles.resourceMark} aria-hidden="true">
          {resource.mark}
        </div>
        <div className={styles.cardIdentity}>
          <span className={styles.organization}>{resource.organization}</span>
          <span className={styles.sourceBadge}>{sourceLabel}</span>
        </div>
        {resource.updated ? (
          <span className={styles.updatedBadge}>
            {copy.updatedBadge} {resource.updated}
          </span>
        ) : null}
      </div>

      <Heading as="h3" className={styles.resourceTitle}>
        {resource.title}
      </Heading>
      <p className={styles.resourceDescription}>{description}</p>

      <div
        className={styles.metadata}
        aria-label={isChinese ? "资源信息" : "Resource metadata"}
      >
        <span>{categoryLabel}</span>
        <span>{kind}</span>
        <span>{level}</span>
        <span>{localize(resource.access, isChinese)}</span>
        {resource.openSource ? <span>{copy.openSource}</span> : null}
      </div>

      <ul
        className={styles.tagList}
        aria-label={isChinese ? "主题标签" : "Topic tags"}
      >
        {resource.tags.map((tag) => (
          <li key={tag}>{tag}</li>
        ))}
      </ul>

      <Link
        className={styles.resourceAction}
        to={resource.url}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`${copy.openResource}: ${resource.title} (${copy.newTab})`}
      >
        <span>{copy.openResource}</span>
        <span aria-hidden="true">↗</span>
      </Link>
    </article>
  );
}

export default function ResourcesPage() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? pageCopy.zh : pageCopy.en;
  const [query, setQuery] = React.useState("");
  const [category, setCategory] = React.useState("all");
  const [level, setLevel] = React.useState("all");

  const categoryCounts = React.useMemo(() => {
    return learningResources.reduce((counts, resource) => {
      counts[resource.category] = (counts[resource.category] || 0) + 1;
      return counts;
    }, {});
  }, []);

  const visibleResources = React.useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();

    return learningResources.filter((resource) => {
      const matchesCategory =
        category === "all" || resource.category === category;
      const matchesLevel = level === "all" || resource.level === level;
      if (!matchesCategory || !matchesLevel) return false;
      if (!normalizedQuery) return true;

      const categoryItem = resourceCategories.find(
        (item) => item.id === resource.category
      );
      const searchableText = [
        resource.title,
        resource.organization,
        localize(resource.description, isChinese),
        localize(categoryItem.label, isChinese),
        localize(resourceKinds[resource.kind], isChinese),
        localize(resourceLevels[resource.level], isChinese),
        ...resource.tags,
      ]
        .join(" ")
        .toLocaleLowerCase();

      return searchableText.includes(normalizedQuery);
    });
  }, [category, isChinese, level, query]);

  const visibleSections = React.useMemo(
    () =>
      resourceCategories
        .filter((item) => category === "all" || category === item.id)
        .map((item) => ({
          category: item,
          bands: levelOrder
            .map((levelId) => ({
              level: levelId,
              items: visibleResources.filter(
                (resource) =>
                  resource.category === item.id && resource.level === levelId
              ),
            }))
            .filter((band) => band.items.length > 0),
        }))
        .filter((section) => section.bands.length > 0),
    [category, visibleResources]
  );

  const hasFilters = query || category !== "all" || level !== "all";

  function clearFilters() {
    setQuery("");
    setCategory("all");
    setLevel("all");
  }

  function scrollToCatalog(categoryId) {
    window.requestAnimationFrame(() => {
      const target =
        (categoryId && categoryId !== "all"
          ? document.getElementById(`resource-${categoryId}`)
          : null) || document.getElementById("resource-catalog");
      target?.scrollIntoView({ block: "start", behavior: "smooth" });
    });
  }

  function selectCategory(categoryId) {
    setCategory(categoryId);
    setQuery("");
    scrollToCatalog(categoryId);
  }

  function openLearningPath(pathCategory) {
    setCategory(pathCategory);
    setLevel("all");
    setQuery("");
    scrollToCatalog(pathCategory);
  }

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
                {copy.heroTitle}
              </Heading>
              <p className={styles.heroDescription}>{copy.heroDescription}</p>
            </div>

            <dl className={styles.heroStats}>
              <div>
                <dt>{copy.statResources}</dt>
                <dd>{learningResources.length}</dd>
              </div>
              <div>
                <dt>{copy.statTopics}</dt>
                <dd>{resourceCategories.length}</dd>
              </div>
              <div>
                <dt>{copy.statSourcesLabel}</dt>
                <dd className={styles.textStat}>{copy.statSources}</dd>
              </div>
              <div>
                <dt>{copy.statVerifiedLabel}</dt>
                <dd className={styles.textStat}>{copy.statVerified}</dd>
              </div>
            </dl>
          </header>

          <section
            className={styles.pathsSection}
            aria-labelledby="paths-title"
          >
            <div className={styles.sectionHeading}>
              <div>
                <span className={styles.sectionEyebrow}>
                  {copy.pathsEyebrow}
                </span>
                <Heading
                  as="h2"
                  id="paths-title"
                  className={styles.sectionTitle}
                >
                  {copy.pathsTitle}
                </Heading>
              </div>
              <p>{copy.pathsDescription}</p>
            </div>

            <div className={styles.pathGrid}>
              {learningPaths.map((path) => (
                <article
                  key={path.id}
                  className={clsx(styles.pathCard, styles[path.category])}
                >
                  <div className={styles.pathHeader}>
                    <span className={styles.pathIndex}>{path.index}</span>
                    <span className={styles.pathEyebrow}>
                      {localize(path.eyebrow, isChinese)}
                    </span>
                  </div>
                  <Heading as="h3" className={styles.pathTitle}>
                    {localize(path.title, isChinese)}
                  </Heading>
                  <p className={styles.pathDescription}>
                    {localize(path.description, isChinese)}
                  </p>
                  <ol className={styles.pathSteps}>
                    {path.steps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ol>
                  <button
                    type="button"
                    className={styles.pathAction}
                    onClick={() => openLearningPath(path.category)}
                  >
                    <span>{copy.openPath}</span>
                    <span aria-hidden="true">↓</span>
                  </button>
                </article>
              ))}
            </div>
          </section>

          <section
            id="resource-catalog"
            className={styles.catalogSection}
            aria-labelledby="catalog-title"
          >
            <div className={styles.sectionHeading}>
              <div>
                <span className={styles.sectionEyebrow}>
                  {copy.catalogEyebrow}
                </span>
                <Heading
                  as="h2"
                  id="catalog-title"
                  className={styles.sectionTitle}
                >
                  {copy.catalogTitle}
                </Heading>
              </div>
              <p>{copy.catalogDescription}</p>
            </div>

            <div className={styles.catalogLayout}>
              <aside className={styles.categoryRail}>
                <div className={styles.railHeading}>
                  <strong>{copy.browseTitle}</strong>
                  <span>{copy.browseHint}</span>
                </div>

                <nav
                  className={styles.categoryList}
                  aria-label={copy.categoryNavLabel}
                >
                  <button
                    type="button"
                    className={clsx(
                      styles.railButton,
                      category === "all" && styles.railButtonActive
                    )}
                    aria-pressed={category === "all"}
                    onClick={() => selectCategory("all")}
                  >
                    <span className={styles.railText}>
                      <strong>{copy.allTopics}</strong>
                    </span>
                    <span className={styles.railCount}>
                      {learningResources.length}
                    </span>
                  </button>

                  {resourceGroups.map((group) => (
                    <div key={group.id} className={styles.railGroup}>
                      <p className={styles.railGroupLabel}>
                        <strong>{localize(group.label, isChinese)}</strong>
                        <small>{localize(group.hint, isChinese)}</small>
                      </p>
                      {resourceCategories
                        .filter((item) => item.group === group.id)
                        .map((item) => (
                          <button
                            key={item.id}
                            type="button"
                            className={clsx(
                              styles.railButton,
                              category === item.id && styles.railButtonActive
                            )}
                            aria-pressed={category === item.id}
                            onClick={() => selectCategory(item.id)}
                          >
                            <span className={styles.railText}>
                              <strong>{localize(item.label, isChinese)}</strong>
                            </span>
                            <span className={styles.railCount}>
                              {categoryCounts[item.id] || 0}
                            </span>
                          </button>
                        ))}
                    </div>
                  ))}
                </nav>

                <div className={styles.railCrossLink}>
                  <Link to="/navigator">{copy.navigatorLink} →</Link>
                </div>
              </aside>

              <div className={styles.catalogPanel}>
                <div className={styles.filterPanel} role="search">
                  <div className={styles.filterTopRow}>
                    <label className={styles.searchField}>
                      <span className={styles.srOnly}>{copy.searchLabel}</span>
                      <span className={styles.searchGlyph} aria-hidden="true">
                        ⌕
                      </span>
                      <input
                        type="search"
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        placeholder={copy.searchPlaceholder}
                      />
                    </label>

                    <label className={styles.levelField}>
                      <span>{copy.levelLabel}</span>
                      <select
                        value={level}
                        onChange={(event) => setLevel(event.target.value)}
                      >
                        <option value="all">{copy.allLevels}</option>
                        {Object.entries(resourceLevels).map(([id, label]) => (
                          <option key={id} value={id}>
                            {localize(label, isChinese)}
                          </option>
                        ))}
                      </select>
                    </label>

                    <button
                      type="button"
                      className={styles.clearButton}
                      onClick={clearFilters}
                      disabled={!hasFilters}
                    >
                      {copy.clear}
                    </button>
                  </div>
                </div>

                <div className={styles.resultsBar} aria-live="polite">
                  <span>
                    {copy.showing} <strong>{visibleResources.length}</strong>{" "}
                    {copy.of} {learningResources.length} {copy.resources}
                  </span>
                  <span className={styles.resultsLine} aria-hidden="true" />
                </div>

                {visibleSections.length ? (
                  <div className={styles.sectionList}>
                    {visibleSections.map(({ category: item, bands }) => (
                      <section
                        key={item.id}
                        id={`resource-${item.id}`}
                        className={clsx(
                          styles.topicSection,
                          styles[item.id]
                        )}
                        aria-labelledby={`resource-${item.id}-title`}
                      >
                        <div className={styles.topicHeader}>
                          <div>
                            <Heading
                              as="h3"
                              id={`resource-${item.id}-title`}
                              className={styles.topicTitle}
                            >
                              {localize(item.label, isChinese)}
                            </Heading>
                            <p>{localize(item.description, isChinese)}</p>
                          </div>
                          <span className={styles.topicCount}>
                            {bands.reduce(
                              (total, band) => total + band.items.length,
                              0
                            )}
                          </span>
                        </div>

                        {bands.map((band) => (
                          <div key={band.level} className={styles.levelBand}>
                            <p className={styles.levelBandLabel}>
                              <span className={styles.levelDot} aria-hidden="true" />
                              {localize(resourceLevels[band.level], isChinese)}
                              <span className={styles.levelBandLine} aria-hidden="true" />
                            </p>
                            <div className={styles.resourceGrid}>
                              {band.items.map((resource) => (
                                <ResourceCard
                                  key={resource.id}
                                  resource={resource}
                                  isChinese={isChinese}
                                  copy={copy}
                                />
                              ))}
                            </div>
                          </div>
                        ))}
                      </section>
                    ))}
                  </div>
                ) : (
                  <div className={styles.emptyState}>
                    <span aria-hidden="true">⌕</span>
                    <Heading as="h3">{copy.noResultsTitle}</Heading>
                    <p>{copy.noResultsDescription}</p>
                    <button type="button" onClick={clearFilters}>
                      {copy.clear}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </section>

          <aside className={styles.methodology}>
            <div>
              <span className={styles.sectionEyebrow}>
                {copy.methodologyEyebrow}
              </span>
              <Heading as="h2" className={styles.methodologyTitle}>
                {copy.methodologyTitle}
              </Heading>
              <p>{copy.methodologyDescription}</p>
            </div>
            <ul>
              <li>{copy.methodologyPointOne}</li>
              <li>{copy.methodologyPointTwo}</li>
              <li>{copy.methodologyPointThree}</li>
            </ul>
          </aside>
        </div>
      </main>
    </Layout>
  );
}
