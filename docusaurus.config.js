import { themes as prismThemes } from 'prism-react-renderer';

const config = {
  staticDirectories: ['static'],
  scripts: [],
  customFields: {
    svgConfig: {
      comic1: 'compress_comic1',
      comic2: 'compress_comic2',
      comic3: 'compress_comic3',
    },
  },
  title: 'Liangchao Deng',
  tagline: 'Ph.D. in AI × Mathematics × Computer × Plant',
  favicon: 'img/favicon.ico',
  url: 'https://smiler488.github.io/',
  baseUrl: '/',
  trailingSlash: false,

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['en', 'zh'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],
  plugins: [
    [
      '@docusaurus/plugin-google-gtag',
      {
        trackingID: 'G-X1RTB3QBXE',
        anonymizeIP: true,
      },
    ],
  ],

  organizationName: 'smiler488',
  projectName: 'smiler488.github.io',
  deploymentBranch: 'gh-pages',
  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
    localeConfigs: {
      en: { label: 'English' },
      'zh-Hans': { label: '中文', direction: 'ltr' },
    },
  },

  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          blogSidebarCount: 'ALL',
          blogSidebarTitle: 'All posts',
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    ({
      image: 'img/docusaurus-social-card.jpg',
      metadata: [
        { name: 'algolia-site-verification', content: '59BB444E51EBC712' },
      ],
      // Algolia Search Config Placeholder
      // algolia: {
      //   appId: 'YOUR_APP_ID',
      //   apiKey: 'YOUR_SEARCH_API_KEY',
      //   indexName: 'YOUR_INDEX_NAME',
      // },
      colorMode: {
        defaultMode: 'light',
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
        title: 'Home',
        logo: {
          alt: 'My Site Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Tutorial',
          },
          { to: '/blog', label: 'Research', position: 'left' },
          { to: '/cv', label: 'CV', position: 'left' },
          { to: '/resources', label: 'Resource', position: 'left' },
          {
            to: "/app",
            label: "App",
            position: "left",
          },
          { type: 'localeDropdown', position: 'right' },
          {
            href: 'https://github.com/smiler488',
            label: 'GitHub',
            position: 'right',
            className: 'no-external-icon',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [],  // 清空原有链接，使用自定义Footer组件
        copyright: `Copyright © ${new Date().getFullYear()} Liangchao Deng. All rights reserved.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
