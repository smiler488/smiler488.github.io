// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  staticDirectories: ['static'],
  scripts: [
    {
      src: '/js/auth-flag.js',
      defer: true,
    },
    {
      src: '/js/supabase-config.js',
      defer: true,
    },
  ],
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
  // Set the production url of your site here
  url: 'https://smiler488.github.io/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',
  // Disable trailing slash so asset URLs don't gain an extra `/`
  trailingSlash: false,

  // Enable Mermaid diagrams in Markdown
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  // Activate Mermaid theme plugin
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

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'smiler488', // Usually your GitHub org/user name.
  projectName: 'smiler488.github.io', // Usually your repo name.
  deploymentBranch: 'gh-pages',
  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          blogSidebarCount: 'ALL',      // 改成 'ALL' 或者具体数字
          blogSidebarTitle: 'All posts',// 标题想叫别的也可以
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
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
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      metadata: [
        { name: 'algolia-site-verification', content: '28A5A8DFE7ED2916' },
      ],
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      algolia: {
        appId: 'USQTG5BJ2J',
        apiKey: '8bab45a0143db40aa33bdec65b748753',
        indexName: 'smiler488',
        contextualSearch: true, // 启用上下文搜索
        searchParameters: {},   // 高级搜索参数（暂时留空）
        searchPagePath: 'search',
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
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            to: "/app",
            label: "App",
            position: "left",
          },
          { to: '/auth', label: 'Account', position: 'right' },
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
