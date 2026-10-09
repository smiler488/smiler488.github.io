/**
 * site-index: one local plugin, two jobs (design/DESIGN_SPEC.md §4.3, §9.2).
 *
 * 1. allContentLoaded: exposes a light index of blog posts (title, permalink,
 *    date, layers) as global data, so the /research hub can list the notes
 *    behind each research layer without duplicating titles.
 * 2. postBuild: writes llms.txt and llms-full.txt for the current locale from
 *    the same data files the pages use, so machine-readable facts never drift
 *    from what visitors see.
 */
import fs from "node:fs/promises";
import path from "node:path";
import {
  LAYERS,
  PUBLICATIONS,
  PUBLICATION_KIND,
  doiUrl,
} from "../../src/data/publications.js";
import { PROJECTS } from "../../src/data/projects.js";
import { APP_MANIFEST } from "../../src/data/appManifest.js";
import { cvContent, cvIdentity } from "../../src/data/cvData.js";

const BLOG_PLUGIN = "docusaurus-plugin-content-blog";

function pick(value, locale) {
  if (value && typeof value === "object" && "en" in value) {
    return locale === "zh" ? value.zh ?? value.en : value.en;
  }
  return value;
}

// Strip MDX-only syntax so llms-full.txt reads as plain Markdown.
function toPlainMarkdown(source) {
  return source
    .replace(/^import .*$/gm, "")
    .replace(/^<[A-Z][^>]*\/>\s*$/gm, "")
    .replace(/<!--\s*truncate\s*-->/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

const COPY = {
  en: {
    layers: "Research layers",
    projects: "Research projects",
    publications: "Publications and software",
    tools: "Browser tools (App Lab)",
    toolNote:
      "Free tools that run in the browser; data stays on the visitor's device unless an AI feature is explicitly used with the visitor's own key.",
    notes: "Research notes",
    optional: "Optional",
    cv: "Curriculum vitae",
    privacy: "Privacy and visitor map",
    profiles: "Profiles",
  },
  zh: {
    layers: "研究分层",
    projects: "研究项目",
    publications: "论文与软件",
    tools: "浏览器工具（App Lab）",
    toolNote:
      "免费、在浏览器中运行的工具；除非访客主动用自己的密钥调用 AI 功能，数据都留在访客本机。",
    notes: "研究笔记",
    optional: "可选",
    cv: "个人简历",
    privacy: "隐私与访客地图",
    profiles: "学术主页",
  },
};

export default function siteIndexPlugin(context) {
  const { siteConfig, i18n } = context;
  const locale = i18n.currentLocale === "zh-Hans" ? "zh" : "en";
  const localePrefix =
    i18n.currentLocale === i18n.defaultLocale ? "" : `/${i18n.currentLocale}`;
  const site = siteConfig.url.replace(/\/$/, "");
  const abs = (route) => `${site}${localePrefix}${route}`;
  let notes = [];

  return {
    name: "site-index",

    async allContentLoaded({ allContent, actions }) {
      const blog = allContent[BLOG_PLUGIN]?.default;
      notes = (blog?.blogPosts ?? [])
        .filter((post) => !post.metadata.unlisted)
        .map((post) => ({
          title: post.metadata.title,
          description: post.metadata.description,
          permalink: post.metadata.permalink,
          date: post.metadata.date,
          layers: post.metadata.frontMatter.layers ?? [],
          articleType: post.metadata.frontMatter.article_type ?? null,
          content: post.content,
        }));
      actions.setGlobalData({
        notes: notes.map(({ content, ...rest }) => rest),
      });
    },

    async postBuild({ outDir }) {
      const t = COPY[locale];
      const cv = cvContent[locale];
      const lines = [];
      lines.push(`# ${cv.hero.name} (${cv.hero.secondaryName})`);
      lines.push("");
      const sep = locale === "zh" ? "，" : ", ";
      const stop = locale === "zh" ? "。" : ". ";
      lines.push(
        `> ${cv.hero.role}${sep}${cv.hero.institution}${stop}${pick(
          {
            en: "Research program: making plant science measurable with AI, from sensing crops to designing crops.",
            zh: "研究主张：用 AI 让植物科学可度量——从感知作物，走向设计作物。",
          },
          locale
        )}`
      );
      lines.push("");
      lines.push(cv.hero.summary);
      lines.push("");
      lines.push(
        `${t.profiles}: [ORCID](${cvIdentity.orcid}) · [Google Scholar](${cvIdentity.scholar}) · [GitHub](${cvIdentity.github})`
      );
      lines.push("");

      lines.push(`## ${t.layers}`);
      lines.push("");
      for (const layer of LAYERS) {
        const layerName =
          locale === "zh"
            ? `第 ${layer.index} 层 · ${pick(layer.name, locale)}`
            : `Layer ${layer.index} · ${pick(layer.name, locale)}`;
        lines.push(
          `- ${layerName} (${layer.id}): ${pick(layer.summary, locale)}`
        );
      }
      lines.push("");

      lines.push(`## ${t.projects}`);
      lines.push("");
      for (const project of PROJECTS) {
        lines.push(
          `- [${pick(project.title, locale)}](${abs(
            `/research/${project.id}`
          )}): ${pick(project.finding, locale)} [${project.layers.join(", ")}]`
        );
      }
      lines.push("");

      lines.push(`## ${t.publications}`);
      lines.push("");
      for (const pub of PUBLICATIONS) {
        lines.push(
          `- [${pub.title}](${doiUrl(pub.doi)}): ${pub.authors.join(", ")} (${
            pub.year
          }). ${pub.venue}. ${PUBLICATION_KIND[pub.type][locale]}. DOI ${
            pub.doi
          }. [${pub.layers.join(", ")}]`
        );
      }
      lines.push("");

      lines.push(`## ${t.tools}`);
      lines.push("");
      lines.push(t.toolNote);
      lines.push("");
      for (const app of APP_MANIFEST) {
        const layers = app.layers?.length ? ` [${app.layers.join(", ")}]` : "";
        lines.push(
          `- [${pick(app.name, locale)}](${abs(app.route)}): ${pick(
            app.description,
            locale
          )}${layers}`
        );
      }
      lines.push("");

      const researchNotes = notes.filter((note) => note.layers.length);
      lines.push(`## ${t.notes}`);
      lines.push("");
      for (const note of researchNotes) {
        lines.push(
          `- [${note.title}](${site}${note.permalink}): ${
            note.description
          } [${note.layers.join(", ")}]`
        );
      }
      lines.push("");

      lines.push(`## ${t.optional}`);
      lines.push("");
      lines.push(`- [${t.cv}](${abs("/cv")})`);
      lines.push(`- [${t.privacy}](${abs("/privacy")})`);
      lines.push("");

      const full = [lines.join("\n")];
      for (const note of researchNotes) {
        full.push(
          `\n---\n\n# ${note.title}\n\nSource: ${site}${
            note.permalink
          }\nLayers: ${note.layers.join(", ")}\n\n${toPlainMarkdown(
            note.content
          )}\n`
        );
      }

      await fs.writeFile(
        path.join(outDir, "llms.txt"),
        lines.join("\n"),
        "utf8"
      );
      await fs.writeFile(
        path.join(outDir, "llms-full.txt"),
        full.join("\n"),
        "utf8"
      );
    },
  };
}
