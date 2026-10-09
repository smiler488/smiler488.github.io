/**
 * Research hub (design/DESIGN_SPEC.md §6.2): the four-layer program as a map
 * of evidence. ?layer=DIG|UND|PRE|DES filters projects, notes and tools.
 */
import React from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import { useHistory, useLocation } from "@docusaurus/router";
import { usePluginData } from "@docusaurus/useGlobalData";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { LAYERS, PUBLICATIONS } from "@site/src/data/publications";
import { PROJECTS, PROJECT_STATUS } from "@site/src/data/projects";
import { APP_MANIFEST } from "@site/src/data/appManifest";
import {
  Chip,
  JsonLd,
  LayerBadges,
  PageHero,
  SectionHeader,
  Stats,
  useIsChinese,
  useLocalize,
} from "@site/src/components/ds";
import styles from "./styles.module.css";

const COPY = {
  en: {
    pageTitle: "Research",
    pageDescription:
      "A four-layer research program in AI for plant phenotyping and crop modeling: digitize, understand, predict and design crops, with the evidence behind each layer.",
    eyebrow: "Research program",
    title: "From sensing crops to designing crops.",
    lead: "Four layers turn multi-source observations into a crop that is measurable, explainable, predictable and, eventually, designable. Each layer below links to the projects, papers, notes and tools that support it.",
    statProjects: "projects",
    statPublications: "publications and software",
    statNotes: "research notes",
    filterLabel: "Filter by layer",
    all: "All layers",
    layersEyebrow: "Architecture",
    layersTitle: "Four layers of crop intelligence",
    layersDescription:
      "My PhD built Layers I–II. My postdoc focuses on the jump from II to III: making crop state not only observable and explainable, but projectable.",
    layerCounts: (p, n, t) => `${p} projects · ${n} notes · ${t} tools`,
    projectsEyebrow: "Projects",
    projectsTitle: "Selected projects",
    projectsEmpty:
      "No project page in this layer yet. The notes below cover the work so far.",
    notesEyebrow: "Notes",
    notesTitle: "Research notes",
    notesEmpty: "No research notes in this layer yet.",
    toolsTitle: "Tools in the Lab",
    morePublications: "All publications",
    moreNotes: "All notes",
    moreLab: "Open the Lab",
  },
  zh: {
    pageTitle: "研究",
    pageDescription:
      "面向植物表型与作物模型的四层 AI 研究计划：数字化、理解、预测与设计作物，并列出每一层背后的证据。",
    eyebrow: "研究计划",
    title: "从感知作物，走向设计作物。",
    lead: "四个层次把多源观测转化为可度量、可解释、可预测、最终可设计的作物。下面每一层都链接到支撑它的项目、论文、笔记与工具。",
    statProjects: "个项目",
    statPublications: "篇论文与软件",
    statNotes: "篇研究笔记",
    filterLabel: "按层筛选",
    all: "全部层次",
    layersEyebrow: "研究架构",
    layersTitle: "作物智能的四个层次",
    layersDescription:
      "博士阶段构建了第 I–II 层；博士后聚焦从 II 到 III 的跃迁：让作物状态不仅可观测、可解释，而且可预测。",
    layerCounts: (p, n, t) => `${p} 个项目 · ${n} 篇笔记 · ${t} 个工具`,
    projectsEyebrow: "项目",
    projectsTitle: "代表性项目",
    projectsEmpty: "这一层暂时还没有项目页，下面的笔记记录了目前的工作。",
    notesEyebrow: "笔记",
    notesTitle: "研究笔记",
    notesEmpty: "这一层暂时还没有研究笔记。",
    toolsTitle: "实验室中的工具",
    morePublications: "全部论文",
    moreNotes: "全部笔记",
    moreLab: "进入实验室",
  },
};

const LAYER_IDS = LAYERS.map((layer) => layer.id);

function useLayerFilter() {
  const location = useLocation();
  const history = useHistory();
  const param = new URLSearchParams(location.search).get("layer");
  const active = LAYER_IDS.includes(param) ? param : null;
  const setActive = (id) => {
    const search = id ? `?layer=${id}` : "";
    history.replace({ ...location, search });
  };
  return [active, setActive];
}

function ProjectCard({ project, localize }) {
  const cover = useBaseUrl(project.cover);
  return (
    <Link className={styles.projectCard} to={`/research/${project.id}`}>
      <div className={styles.projectCover}>
        <img src={cover} alt="" loading="lazy" />
      </div>
      <div className={styles.projectMeta}>
        <LayerBadges ids={project.layers} link={false} />
        <span>
          {localize(PROJECT_STATUS[project.status])} · {project.year}
        </span>
      </div>
      <Heading as="h3" className={styles.projectTitle}>
        {localize(project.title)}
      </Heading>
      <p className={styles.projectFinding}>{localize(project.finding)}</p>
    </Link>
  );
}

export default function ResearchHub() {
  const isChinese = useIsChinese();
  const localize = useLocalize();
  const copy = isChinese ? COPY.zh : COPY.en;
  const { notes = [] } = usePluginData("site-index") ?? {};
  const [active, setActive] = useLayerFilter();
  const siteUrl = useBaseUrl("/", { absolute: true });

  const researchNotes = notes
    .filter((note) => note.layers?.length)
    .sort((a, b) => new Date(b.date) - new Date(a.date));
  const inLayer = (item) => !active || item.layers?.includes(active);
  const projects = PROJECTS.filter(inLayer);
  const visibleNotes = researchNotes.filter(inLayer);
  const tools = APP_MANIFEST.filter(
    (app) => app.layers?.length && inLayer(app)
  );

  const countFor = (id) => ({
    projects: PROJECTS.filter((p) => p.layers.includes(id)).length,
    notes: researchNotes.filter((n) => n.layers.includes(id)).length,
    tools: APP_MANIFEST.filter((a) => a.layers?.includes(id)).length,
  });

  const dateFormatter = new Intl.DateTimeFormat(isChinese ? "zh-CN" : "en", {
    year: "numeric",
    month: "short",
    timeZone: "UTC",
  });

  return (
    <Layout title={copy.pageTitle} description={copy.pageDescription}>
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "CollectionPage",
          name: copy.pageTitle,
          description: copy.pageDescription,
          url: `${siteUrl}research`,
          hasPart: PROJECTS.map((p) => ({
            "@type": "ResearchProject",
            name: localize(p.title),
            url: `${siteUrl}research/${p.id}`,
          })),
        }}
      />
      <main className={styles.page}>
        <div className={styles.shell}>
          <PageHero eyebrow={copy.eyebrow} title={copy.title} lead={copy.lead}>
            <Stats
              items={[
                { value: PROJECTS.length, label: copy.statProjects },
                { value: PUBLICATIONS.length, label: copy.statPublications },
                { value: researchNotes.length, label: copy.statNotes },
              ]}
            />
          </PageHero>

          <div
            className={styles.filterBar}
            role="group"
            aria-label={copy.filterLabel}
          >
            <Chip active={!active} onClick={() => setActive(null)}>
              {copy.all}
            </Chip>
            {LAYERS.map((layer) => (
              <Chip
                key={layer.id}
                active={active === layer.id}
                onClick={() => setActive(active === layer.id ? null : layer.id)}
              >
                {layer.index} · {localize(layer.name)}
              </Chip>
            ))}
          </div>

          <section className={styles.section} aria-labelledby="layers-title">
            <SectionHeader
              id="layers-title"
              eyebrow={copy.layersEyebrow}
              title={copy.layersTitle}
              description={copy.layersDescription}
            />
            <div className={styles.layerGrid}>
              {LAYERS.map((layer) => {
                const c = countFor(layer.id);
                const isActive = active === layer.id;
                return (
                  <button
                    key={layer.id}
                    type="button"
                    className={styles.layerCard}
                    aria-pressed={isActive}
                    data-active={isActive || undefined}
                    data-dim={(active && !isActive) || undefined}
                    onClick={() => setActive(isActive ? null : layer.id)}
                  >
                    <span className={styles.layerIndex}>{layer.index}</span>
                    <strong>{localize(layer.name)}</strong>
                    <span className={styles.layerSummary}>
                      {localize(layer.summary)}
                    </span>
                    <span className={styles.layerCounts}>
                      {copy.layerCounts(c.projects, c.notes, c.tools)}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className={styles.section} aria-labelledby="projects-title">
            <SectionHeader
              id="projects-title"
              eyebrow={copy.projectsEyebrow}
              title={copy.projectsTitle}
            />
            {projects.length ? (
              <div className={styles.projectGrid}>
                {projects.map((project) => (
                  <ProjectCard
                    key={project.id}
                    project={project}
                    localize={localize}
                  />
                ))}
              </div>
            ) : (
              <p className={styles.empty}>{copy.projectsEmpty}</p>
            )}
            <div className={styles.moreRow}>
              <Link className={styles.moreLink} to="/publications">
                {copy.morePublications}
              </Link>
            </div>
          </section>

          <section className={styles.section} aria-labelledby="notes-title">
            <SectionHeader
              id="notes-title"
              eyebrow={copy.notesEyebrow}
              title={copy.notesTitle}
            />
            {visibleNotes.length ? (
              <ol className={styles.noteList}>
                {visibleNotes.map((note) => (
                  <li key={note.permalink}>
                    <time dateTime={note.date}>
                      {dateFormatter.format(new Date(note.date))}
                    </time>
                    <div>
                      <Link to={note.permalink}>{note.title}</Link>
                      <p>{note.description}</p>
                    </div>
                    <LayerBadges ids={note.layers} link={false} />
                  </li>
                ))}
              </ol>
            ) : (
              <p className={styles.empty}>{copy.notesEmpty}</p>
            )}
            <div className={styles.moreRow}>
              <Link className={styles.moreLink} to="/blog">
                {copy.moreNotes}
              </Link>
            </div>
          </section>

          {tools.length > 0 && (
            <section className={styles.section} aria-labelledby="tools-title">
              <SectionHeader id="tools-title" title={copy.toolsTitle} />
              <div className={styles.toolRow}>
                {tools.map((app) => (
                  <Chip key={app.id} to={app.route}>
                    {localize(app.shortName ?? app.name)}
                  </Chip>
                ))}
              </div>
              <div className={styles.moreRow}>
                <Link className={styles.moreLink} to="/app">
                  {copy.moreLab}
                </Link>
              </div>
            </section>
          )}
        </div>
      </main>
    </Layout>
  );
}
