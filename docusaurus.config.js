import { execSync } from "node:child_process";
import { createRequire } from "node:module";
import { themes as prismThemes } from "prism-react-renderer";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const require = createRequire(import.meta.url);

// $…$ and $$…$$ in Markdown render as KaTeX (blog, docs and MDX pages).
const math = { remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] };

// Short commit hash, stamped into tool parameter records (DESIGN_SPEC §8.2).
function buildStamp() {
  try {
    return execSync("git rev-parse --short HEAD").toString().trim();
  } catch {
    return null;
  }
}

const config = {
  staticDirectories: ["static"],
  scripts: [],
  customFields: {
    build: buildStamp(),
    // Supabase anon keys are public client identifiers. Real authorization must
    // still be enforced with Row Level Security in Supabase.
    supabaseUrl: process.env.SUPABASE_URL || null,
    supabaseAnonKey: process.env.SUPABASE_ANON_KEY || null,
    svgConfig: {
      comic1: "compress_comic1",
      comic2: "compress_comic2",
      comic3: "compress_comic3",
    },
  },
  title: "Liangchao Deng",
  tagline: "AI4Plant · Plant Phenotyping & Crop Modeling",
  favicon: "img/favicon.ico",
  url: "https://smiler488.github.io",
  baseUrl: "/",
  trailingSlash: false,

  // One-time reset of a light/dark choice saved by the old two-state toggle,
  // so every visitor starts on "follow the system" (DESIGN_SPEC D-020). Runs
  // in <head>, before Docusaurus reads the stored theme.
  headTags: [
    {
      tagName: "script",
      attributes: {},
      innerHTML:
        'try{var k="theme-system-default";if(!localStorage.getItem(k)){localStorage.removeItem("theme");localStorage.setItem(k,"1")}}catch(e){}',
    },
  ],

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },
  themes: [
    "@docusaurus/theme-mermaid",
    [
      "@easyops-cn/docusaurus-search-local",
      {
        hashed: true,
        language: ["en", "zh"],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],
  // No first-party analytics plugin is enabled. The footer's third-party
  // visitor map image runs in an opaque sandbox and is disclosed at /privacy.
  plugins: [
    // Blog index for the /research hub + llms.txt generation (DESIGN_SPEC §9.2).
    "./plugins/site-index",
  ],

  organizationName: "smiler488",
  projectName: "smiler488.github.io",
  deploymentBranch: "gh-pages",
  onBrokenLinks: "throw",

  i18n: {
    defaultLocale: "en",
    locales: ["en", "zh-Hans"],
    localeConfigs: {
      en: { label: "English" },
      "zh-Hans": { label: "中文", direction: "ltr" },
    },
  },

  presets: [
    [
      "classic",
      {
        docs: {
          ...math,
          sidebarPath: "./sidebars.js",
          editUrl:
            "https://github.com/smiler488/smiler488.github.io/tree/master/",
        },
        blog: {
          ...math,
          showReadingTime: true,
          blogTitle: "Research & Engineering Notes",
          blogDescription:
            "Research notes on artificial intelligence, plant phenotyping, imaging, and reproducible scientific computing.",
          postsPerPage: 9,
          blogSidebarCount: 0,
          feedOptions: {
            type: ["rss", "atom"],
            title: "Liangchao Deng · Research & Engineering Notes",
            description:
              "Practical research notes on AI, plant phenotyping, imaging, and scientific software.",
            xslt: true,
          },
          editUrl:
            "https://github.com/smiler488/smiler488.github.io/tree/master/",
          onInlineTags: "warn",
          onInlineAuthors: "warn",
          onUntruncatedBlogPosts: "warn",
        },
        pages: math,
        theme: {
          // KaTeX fonts download only on pages that contain a formula
          customCss: [
            "./src/css/tokens.css",
            "./src/css/custom.css",
            require.resolve("katex/dist/katex.min.css"),
          ],
        },
        sitemap: {
          // Demo and utility routes should not be advertised to crawlers.
          ignorePatterns: [
            "/search",
            "/auth",
            "/hologram",
            "/app/maze",
            "/zh-Hans/search",
            "/zh-Hans/auth",
            "/zh-Hans/hologram",
            "/zh-Hans/app/maze",
          ],
        },
      },
    ],
  ],

  themeConfig: {
    // Link-preview card per locale (1200×630); the old Docusaurus default
    // card stays in static/ only so cached previews don't 404.
    image:
      process.env.DOCUSAURUS_CURRENT_LOCALE === "zh-Hans"
        ? "img/social-card-zh.jpg"
        : "img/social-card.jpg",
    metadata: [
      { name: "algolia-site-verification", content: "59BB444E51EBC712" },
      {
        name: "google-site-verification",
        content: "1F_blYF74vUYiqjx5hRpaHBnAHqS5MMkVDRA_utuAxU",
      },
    ],
    // Algolia Search Config Placeholder
    // algolia: {
    //   appId: 'YOUR_APP_ID',
    //   apiKey: 'YOUR_SEARCH_API_KEY',
    //   indexName: 'YOUR_INDEX_NAME',
    // },
    colorMode: {
      defaultMode: "light",
      disableSwitch: false,
      // Three-state toggle: system → light → dark. "system" follows the OS
      // setting; defaultMode only applies where the OS reports no preference.
      respectPrefersColorScheme: true,
    },

    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
    navbar: {
      hideOnScroll: true,
      title: "Liangchao Deng",
      logo: {
        alt: "Liangchao Deng · AI4Plant",
        src: "img/logo.svg",
      },
      items: [
        // Primary navigation: research identity only (DESIGN_SPEC §4.1).
        { to: "/research", label: "Research", position: "left" },
        { to: "/app", label: "Lab", position: "left" },
        { to: "/blog", label: "Notes", position: "left" },
        { to: "/cv", label: "About", position: "left" },
        // Utility pages sit with the utility icons on the right (D-021); the
        // footer, the Lab hub and the About page keep their links too.
        {
          type: "dropdown",
          label: "Toolbox",
          position: "right",
          className: "navbar-toolbox",
          items: [
            { to: "/resources", label: "Learning resources" },
            { to: "/navigator", label: "Navigator" },
            { to: "/mpicks", label: "mPicks" },
          ],
        },
        { type: "localeDropdown", position: "right" },
        {
          href: "https://github.com/smiler488",
          label: "GitHub",
          position: "right",
          className: "no-external-icon",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [], // 清空原有链接，使用自定义Footer组件
      copyright: `Copyright © ${new Date().getFullYear()} Liangchao Deng. All rights reserved.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;
