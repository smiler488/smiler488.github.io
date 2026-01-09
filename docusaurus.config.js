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
  themes: ['@docusaurus/theme-mermaid'],
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
      navbar: {
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
          { to: '/blog', label: 'Blog', position: 'left' },
          { to: '/cv', label: 'CV', position: 'left' },
          { to: '/resources', label: 'Resources', position: 'left' },
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
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'CN Community',
            items: [
              {
                label: 'Bilibili',
                href: 'https://space.bilibili.com/16062789',
              },
              {
                label: 'Douyin',
                href: 'https://v.douyin.com/-1moIAdEYpg/ 4@5.com :2pm',
              },
              {
                label: 'Weibo',
                href: 'https://m.weibo.cn/profile/5283742028',
              },
              {
                label: 'WeChat Offical',
                href: 'https://mp.weixin.qq.com/s/JPLLGnM6fwT8XpBdfoXKNA',
              },
            ],
          },
          {
            title: 'EN Community',
            items: [
              {
                label: 'YouTube',
                href: 'https://www.youtube.com/channel/UCmz7DQ3nEPRxj4rvEQUCvAg',
              },
              {
                label: 'TikTok',
                href: 'https://www.tiktok.com/@smiler488tt',
              },
              {
                label: 'X',
                href: 'https://x.com/smiler488',
              },
              {
                label: 'Reddit',
                href: 'https://www.reddit.com/user/smiler488/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'LinkedIn',
                href: 'https://www.linkedin.com/in/liangchao-deng-7b420b269/',
              },
              {
                label: 'HuggingFace',
                href: 'https://huggingface.co/smiler488',
              },
              {
                label: 'Bluesky',
                href: 'https://bsky.app/profile/smiler488.bsky.social',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Liangchao Deng. All rights reserved.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
