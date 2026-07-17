---
slug: personal-website-docusaurus-github-pages
title: "用 Docusaurus 和 GitHub Pages 搭建并发布个人网站"
description: "一套注重版本的工作流：创建 Docusaurus 作品集，正确配置 GitHub Pages，并维护可靠的部署。"
authors: [liangchao]
category: 开发者工具
article_type: 技术指南
tags: [web-development, git, reproducible-research]
image: /img/blog-default.jpg
---

## 项目概述

本指南用 Docusaurus 搭建个人作品集，并将生成的静态站点发布到 GitHub Pages。它涵盖导致大多数部署失败的两个细节：选择正确的 `baseUrl`，以及把源码分支与生成的 `gh-pages` 分支分开。

- **适用场景：** 作品集、项目文档、论文列表和技术笔记
- **结果：** 一个版本可控的站点，本地与托管构建均可重复
- **当前基线：** Docusaurus 3 需要 Node.js 20 或更新版本

<!-- truncate -->

## 开始之前

安装或准备：

- Node.js 20 或更新版本以及 npm
- Git
- 一个 GitHub 账号
- 一个代码编辑器

检查本地工具：

```bash
node --version
npm --version
git --version
```

下面的命令使用 npm。在整个项目中保持使用同一个包管理器及其锁文件。

## 1. 创建站点

使用官方项目生成器：

```bash
npx create-docusaurus@latest my-portfolio classic
cd my-portfolio
npm install
npm run start
```

开发服务器通常在 `http://localhost:3000` 打开。对 Markdown、React 组件和 CSS 的修改会通过热重载呈现。

classic 模板包含：

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

没有默认的 `npm run new blog` 命令。创建一个诸如 `blog/2026-07-16-field-workflow.md` 的文件并添加合法的 frontmatter 即可。

## 2. 确定 GitHub Pages 地址

仓库名决定了公开路径。

| 站点类型          | 仓库                   | 公开 URL                                     | `baseUrl`        |
| ------------------------- | ---------------------- | -------------------------------------------- | ---------------- |
| 用户或组织站点    | `<username>.github.io` | `https://<username>.github.io/`              | `/`              |
| 项目站点          | `my-portfolio`         | `https://<username>.github.io/my-portfolio/` | `/my-portfolio/` |

为其中一个目标配置 `docusaurus.config.js`。下面这个用户站点示例保留了干净的根 URL：

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

对于项目站点，把 `projectName` 改为仓库名，并把 `baseUrl` 设为 `/repository-name/`。不要在部署工作流中把这个值留给一个未设置的环境变量。

## 3. 添加内容和身份信息

优先提供访问者需要的信息：

- 一段简短的研究或职业简介
- 当前项目及其具体成果
- 论文、数据集、软件和可复现的工作流
- 目的明确的联系方式
- 一个简洁的博客，文章带日期并持续维护

把可下载的资源放在 `static/` 下。例如 `static/files/cv.pdf` 可通过 `/files/cv.pdf` 访问。

使用博客作者注册表，而不是在每篇文章里重复写职称。更新一条作者记录，可以避免旧的身份文字残留在归档文章中。

## 4. 验证生产构建

发布前先在本地运行同样的构建：

```bash
npm run build
npm run serve
```

预览使用生成的 `build/` 目录。在部署之前，解决失效链接、非法 frontmatter 和缺失资源等问题。

## 5. 连接源码仓库

创建一个空的 GitHub 仓库，然后连接本地项目：

```bash
git init
git add .
git commit -m "feat: create personal website"
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

源码分支包含可编辑的代码。`gh-pages` 分支应只包含生成的部署文件。

## 6. 用 GitHub Actions 部署

创建 `.github/workflows/deploy.yml`：

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

如果源码分支是 `master`，相应更新工作流触发条件。在 **Settings → Pages** 中，选择 `gh-pages` 分支及其根目录作为发布源。

对于受控的手动部署，Docusaurus 还提供了：

```bash
npx docusaurus deploy
```

请一致地使用一种部署方法。项目专用的 `npm run deploy` 脚本可以封装构建和发布命令。

## 7. 维护工作流

每次更新：

1. 当改动较大时，在功能分支上编辑；
2. 运行生产构建；
3. 在桌面端和移动端审查生成的页面；
4. 提交源码改动；
5. 推送已配置的源码分支并监控部署任务。

定期查阅依赖的发布说明，再将所有 `@docusaurus/*` 包升级到同一版本。

## 故障排查

### 样式或资源返回 404

确认 `url`、`baseUrl` 和仓库名描述的是同一托管路径。项目站点却用 `baseUrl: '/'` 部署，是最常见的原因。

### 某个路由能通过导航访问，但在公开 URL 上打不开

Docusaurus 生成的是静态路由，而不是依赖通用的单页应用回退。设置明确的 `trailingSlash` 策略，核对生成的文件，并让 GitHub Pages 从预期的分支发布。

### Action 能构建但无法发布

确认 `permissions: contents: write`、仓库的 Actions 权限、源码分支触发条件以及 Pages 发布分支。

### 自定义域名无法解析

在 GitHub Pages 和 DNS 提供商处都配置自定义域名。子域名通常使用 CNAME 记录；根域名使用受支持的 A、ALIAS 或 ANAME 记录。仅当部署工作流要求该文件包含在每次构建中时，才保留 `static/CNAME`。

## 官方参考资料

- [Docusaurus 安装](https://docusaurus.io/docs/installation)
- [Docusaurus 部署](https://docusaurus.io/docs/deployment)
- [GitHub Pages 文档](https://docs.github.com/pages)
