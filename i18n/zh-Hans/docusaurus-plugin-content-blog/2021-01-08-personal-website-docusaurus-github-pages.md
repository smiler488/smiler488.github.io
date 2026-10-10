---
slug: personal-website-docusaurus-github-pages
title: "用 Docusaurus 和 GitHub Pages 搭建并发布个人网站"
description: "从创建项目、配置 GitHub Pages 地址到自动部署，用 Docusaurus 搭建并长期维护个人网站。"
authors: [liangchao]
category: 开发者工具
article_type: 技术指南
tags: [web-development, git, reproducible-research]
image: /img/blog-default.jpg
---

## 概述

本文介绍如何用 Docusaurus 搭建个人主页，并把生成的静态网站发布到 GitHub Pages。部署失败大多出在两处：`baseUrl` 设置错误，以及源码分支和部署用的 `gh-pages` 分支混在一起。下文会重点说明这两点。

- **适合用来做：** 个人主页、项目文档、论文列表和技术笔记
- **最终效果：** 网站纳入版本管理，本地构建和线上构建结果一致
- **版本要求：** Docusaurus 3 需要 Node.js 20 及以上

<!-- truncate -->

## 准备工作

需要先安装：

- Node.js 20 及以上，以及 npm
- Git
- GitHub 账号
- 代码编辑器

检查本地环境：

```bash
node --version
npm --version
git --version
```

下文统一使用 npm。同一个项目请始终用同一种包管理器，并提交它的锁文件。

## 1. 创建网站

用官方脚手架创建项目：

```bash
npx create-docusaurus@latest my-portfolio classic
cd my-portfolio
npm install
npm run start
```

开发服务器一般运行在 `http://localhost:3000`，修改 Markdown、React 组件或 CSS 后页面会自动刷新。

classic 模板的目录结构如下：

```text
my-portfolio/
├── blog/                  # 带日期的 Markdown 或 MDX 文章
├── docs/                  # 文档和长篇页面
├── src/
│   ├── components/        # 可复用的 React 组件
│   ├── css/custom.css     # 全站样式
│   └── pages/             # 独立路由
├── static/                # 直接复制到构建产物中的文件
├── docusaurus.config.js
├── sidebars.js
└── package.json
```

Docusaurus 没有 `npm run new blog` 这样的命令。新建一篇博客，只需创建一个文件，例如 `blog/2026-07-16-field-workflow.md`，并写好 front matter。

## 2. 确定 GitHub Pages 地址

网站的公开地址由仓库名决定：

| 网站类型 | 仓库名 | 公开地址 | `baseUrl` |
| --- | --- | --- | --- |
| 用户或组织主页 | `<username>.github.io` | `https://<username>.github.io/` | `/` |
| 项目页面 | `my-portfolio` | `https://<username>.github.io/my-portfolio/` | `/my-portfolio/` |

按其中一种情况配置 `docusaurus.config.js`。下面是用户主页的例子，网址就是根路径：

```js
const config = {
  title: "My Portfolio",
  url: "https://<username>.github.io",
  baseUrl: "/",
  organizationName: "<username>",
  projectName: "<username>.github.io",
  deploymentBranch: "gh-pages",
  trailingSlash: false,
  // ...
};

export default config;
```

如果是项目页面，把 `projectName` 改成仓库名，`baseUrl` 设为 `/仓库名/`。这个值请直接写在配置里，不要依赖部署流程中可能没有设置的环境变量。

## 3. 填充内容

先放访客最关心的信息：

- 简短的研究或职业介绍
- 正在做的项目及其具体成果
- 论文、数据集、软件和可复现的流程
- 联系方式，并说明适合联系的事由
- 博客：文章标注日期，并持续更新

可供下载的文件放在 `static/` 下，例如 `static/files/cv.pdf` 的访问地址就是 `/files/cv.pdf`。

作者信息写在博客的作者配置文件中，不要在每篇文章里重复填写职称。这样职位变动时只需改一处，旧文章也会同步更新。

## 4. 本地检查生产构建

发布前先在本地跑一遍正式构建：

```bash
npm run build
npm run serve
```

预览的是生成的 `build/` 目录。失效链接、front matter 格式错误、资源缺失等问题，都应在部署前解决。

## 5. 关联 GitHub 仓库

在 GitHub 上新建一个空仓库，然后把本地项目推送上去：

```bash
git init
git add .
git commit -m "feat: create personal website"
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

源码分支存放可编辑的代码，`gh-pages` 分支只存放构建生成的网站文件。

## 6. 用 GitHub Actions 自动部署

新建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run build
      - name: Publish generated site
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build
          publish_branch: gh-pages
```

如果源码分支叫 `master`，把触发分支改成 `master`。然后在仓库的 **Settings → Pages** 中，把发布来源设为 `gh-pages` 分支的根目录。

如果希望手动控制部署，也可以用 Docusaurus 自带的命令：

```bash
npx docusaurus deploy
```

自动部署和手动部署选一种即可，不要混用。也可以在 `package.json` 中写一个 `npm run deploy` 脚本，把构建和发布合并成一步。

## 7. 日常维护

每次更新网站：

1. 改动较大时，在单独的功能分支上修改；
2. 运行生产构建；
3. 在电脑和手机上分别检查页面；
4. 提交源码改动；
5. 推送到源码分支，并查看部署任务是否成功。

升级依赖前先看发布说明，所有 `@docusaurus/*` 包要升级到同一版本。

## 常见问题

### 样式或图片 404

检查 `url`、`baseUrl` 和仓库名是否对应同一个地址。最常见的原因是项目页面却把 `baseUrl` 设成了 `/`。

### 站内点击能打开，直接访问网址却 404

Docusaurus 为每个路由生成独立的静态页面，并不依赖单页应用的路由回退。请明确设置 `trailingSlash`，检查生成的文件是否存在，并确认 GitHub Pages 的发布分支设置正确。

### Action 构建成功但发布失败

检查工作流中是否有 `permissions: contents: write`、仓库是否允许 Actions 写入、触发分支是否正确，以及 Pages 的发布分支设置。

### 自定义域名无法访问

自定义域名需要在 GitHub Pages 和域名服务商两边都配置。子域名一般用 CNAME 记录，根域名用服务商支持的 A、ALIAS 或 ANAME 记录。只有当部署流程要求每次构建都包含 `CNAME` 文件时，才需要保留 `static/CNAME`。

## 官方文档

- [Docusaurus 安装](https://docusaurus.io/docs/installation)
- [Docusaurus 部署](https://docusaurus.io/docs/deployment)
- [GitHub Pages 文档](https://docs.github.com/pages)
