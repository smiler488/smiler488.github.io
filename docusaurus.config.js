import { themes as prismThemes } from "prism-react-renderer";

const config = {
  staticDirectories: ["static"],
  scripts: [],
  customFields: {
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
  tagline: "Postdoctoral Researcher · AI for Plant Phenotyping & Crop Modeling",
  favicon: "img/favicon.ico",
  url: "https://smiler488.github.io",
  baseUrl: "/",
  trailingSlash: false,

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
  plugins: [],

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
          sidebarPath: "./sidebars.js",
          editUrl:
            "https://github.com/smiler488/smiler488.github.io/tree/master/",
        },
        blog: {
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
        theme: {
          customCss: "./src/css/custom.css",
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
    image: "img/docusaurus-social-card.jpg",
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
      title: "AzureAxion",
      logo: {
        alt: "AzureAxion — Liangchao Deng",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Tutorial",
        },
        { to: "/blog", label: "Research", position: "left" },
        { to: "/cv", label: "CV", position: "left" },
        { to: "/resources", label: "Resource", position: "left" },
        { to: "/navigator", label: "Navigator", position: "left" },
        {
          to: "/app",
          label: "App",
          position: "left",
        },
        {
          to: "/mpicks",
          label: "mPicks",
          position: "left",
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
