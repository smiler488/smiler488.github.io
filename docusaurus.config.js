// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  configureWebpack: () => ({
    module: {
      rules: [
        {
          test: /compress_comic1\.svg$/,
          issuer: /\.[jt]sx?$/,
          use: [
            {
              loader: require.resolve('@svgr/webpack'),
              options: {
                svgo: true,
                svgoConfig: {
                  plugins: [
                    { name: 'preset-default', params: { overrides: { removeViewBox: false } } },
                    { name: 'prefixIds', params: { prefix: 'compress_comic1-' } },
                  ],
                },
              },
            },
          ],
        },
        {
          test: /compress_comic2\.svg$/,
          issuer: /\.[jt]sx?$/,
          use: [
            {
              loader: require.resolve('@svgr/webpack'),
              options: {
                svgo: true,
                svgoConfig: {
                  plugins: [
                    { name: 'preset-default', params: { overrides: { removeViewBox: false } } },
                    { name: 'prefixIds', params: { prefix: 'compress_comic2-' } },
                  ],
                },
              },
            },
          ],
        },
        {
          test: /compress_comic3\.svg$/,
          issuer: /\.[jt]sx?$/,
          use: [
            {
              loader: require.resolve('@svgr/webpack'),
              options: {
                svgo: true,
                svgoConfig: {
                  plugins: [
                    { name: 'preset-default', params: { overrides: { removeViewBox: false } } },
                    { name: 'prefixIds', params: { prefix: 'compress_comic3-' } },
                  ],
                },
              },
            },
          ],
        },
      ],
    },
  }),
  title: 'Liangchao Deng',
  tagline: 'PhD in Crop Science (Interdisciplinary: Mathematics × Computer Science × Plant Phenotyping)',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://smiler488.github.io/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',
  // Explicitly set trailingSlash to avoid GitHub Pages redirect issues
  trailingSlash: true,

  // Enable Mermaid diagrams in Markdown
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  // Activate Mermaid theme plugin
  themes: ['@docusaurus/theme-mermaid'],

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
         // ✅ 在这里添加 App 跳转
           {
            to: "/app",
            label: "App",
            position: "left",
           },
          {
            href: 'https://github.com/smiler488',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'light',
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
