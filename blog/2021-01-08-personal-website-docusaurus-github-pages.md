---
slug: personal-website-docusaurus-github-pages
title: Build and Publish a Personal Website with Docusaurus and GitHub Pages
description: A version-aware workflow for creating a Docusaurus portfolio, configuring GitHub Pages correctly, and maintaining a reliable deployment.
authors: [liangchao]
category: Developer tools
article_type: Technical guide
tags: [web-development, git, reproducible-research]
image: /img/blog-default.jpg
---

## Project overview

This guide builds a personal portfolio with Docusaurus and publishes the generated static site to GitHub Pages. It covers the two details that cause most deployment failures: choosing the correct `baseUrl` and keeping the source branch separate from the generated `gh-pages` branch.

- **Best for:** portfolios, project documentation, publication lists, and technical notes
- **Result:** a version-controlled site with repeatable local and hosted builds
- **Current baseline:** Docusaurus 3 requires Node.js 20 or newer

<!-- truncate -->

## Before you start

Install or create:

- Node.js 20 or newer and npm
- Git
- a GitHub account
- a code editor

Check the local tools:

```bash
node --version
npm --version
git --version
```

The commands below use npm. Keep one package manager and its lockfile throughout the project.

## 1. Create the site

Use the official project generator:

```bash
npx create-docusaurus@latest my-portfolio classic
cd my-portfolio
npm install
npm run start
```

The development server normally opens at `http://localhost:3000`. Changes to Markdown, React components, and CSS appear through hot reload.

The classic template includes:

```text
my-portfolio/
├── blog/                  # dated Markdown or MDX posts
├── docs/                  # documentation and long-form pages
├── src/
│   ├── components/        # reusable React components
│   ├── css/custom.css     # site-wide styling
│   └── pages/             # standalone routes
├── static/                # files copied directly to the build
├── docusaurus.config.js
├── sidebars.js
└── package.json
```

There is no default `npm run new blog` command. Create a file such as `blog/2026-07-16-field-workflow.md` and add valid front matter.

## 2. Decide the GitHub Pages address

The repository name determines the public path.

| Site type                 | Repository             | Public URL                                   | `baseUrl`        |
| ------------------------- | ---------------------- | -------------------------------------------- | ---------------- |
| User or organization site | `<username>.github.io` | `https://<username>.github.io/`              | `/`              |
| Project site              | `my-portfolio`         | `https://<username>.github.io/my-portfolio/` | `/my-portfolio/` |

Configure `docusaurus.config.js` for one target. This user-site example preserves clean root URLs:

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

For a project site, change `projectName` to the repository name and set `baseUrl` to `/repository-name/`. Do not leave this value to an unset environment variable in the deployment workflow.

## 3. Add content and identity

Prioritize information visitors need:

- a short research or professional profile
- current projects with concrete outcomes
- publications, datasets, software, and reproducible workflows
- contact routes with a clear purpose
- a concise blog with dated, maintained articles

Put downloadable assets under `static/`. For example, `static/files/cv.pdf` is available at `/files/cv.pdf`.

Use the blog author registry rather than repeating job titles inside every post. Updating one author record prevents old identity text from remaining in archived articles.

## 4. Verify the production build

Run the same build locally before publishing:

```bash
npm run build
npm run serve
```

The preview uses the generated `build/` directory. Resolve broken links, invalid front matter, and missing assets before deployment.

## 5. Connect the source repository

Create an empty GitHub repository, then connect the local project:

```bash
git init
git add .
git commit -m "feat: create personal website"
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

The source branch contains editable code. The `gh-pages` branch should contain only generated deployment files.

## 6. Deploy with GitHub Actions

Create `.github/workflows/deploy.yml`:

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

If the source branch is `master`, update the workflow trigger accordingly. In **Settings → Pages**, select the `gh-pages` branch and its root directory as the publishing source.

For a controlled manual deployment, Docusaurus also provides:

```bash
npx docusaurus deploy
```

Use one deployment method consistently. A project-specific `npm run deploy` script may wrap the build and publish commands.

## 7. Maintenance workflow

For each update:

1. edit on a feature branch when the change is substantial;
2. run the production build;
3. review the generated page on desktop and mobile;
4. commit the source change;
5. push the configured source branch and monitor the deployment job.

Periodically review dependency release notes before upgrading all `@docusaurus/*` packages to the same version.

## Troubleshooting

### Styles or assets return 404

Confirm that `url`, `baseUrl`, and the repository name describe the same hosting path. A project site deployed with `baseUrl: '/'` is the most common cause.

### A route works through navigation but not at its public URL

Docusaurus generates static routes rather than relying on a generic single-page-app fallback. Set an explicit `trailingSlash` policy, verify the generated files, and keep GitHub Pages publishing from the expected branch.

### The action can build but cannot publish

Confirm `permissions: contents: write`, repository Actions permissions, the source-branch trigger, and the Pages publishing branch.

### A custom domain does not resolve

Configure the custom domain in GitHub Pages and at the DNS provider. Subdomains normally use a CNAME record; apex domains use supported A, ALIAS, or ANAME records. Keep `static/CNAME` only when the deployment workflow requires the file to be included in every build.

## Official references

- [Docusaurus installation](https://docusaurus.io/docs/installation)
- [Docusaurus deployment](https://docusaurus.io/docs/deployment)
- [GitHub Pages documentation](https://docs.github.com/pages)
