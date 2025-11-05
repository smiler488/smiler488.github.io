---
title: Build a Free Personal Website with Docusaurus and GitHub Pages
slug: personal-website-docusaurus-github-pages
description: Step-by-step workflow to launch and maintain a personal portfolio site with Docusaurus, deployed for free on GitHub Pages.
---

# Build a Free Personal Website with Docusaurus and GitHub Pages

## 1. Project Goals

- Launch a modern personal website without hosting costs.
- Showcase projects, publications, and blog posts using Markdown/MDX.
- Automate deployment so every push updates production safely.

---

## 2. Prerequisites

- **Accounts:** GitHub account with SSH or HTTPS access configured.
- **Runtime:** Node.js ≥ 18 and npm ≥ 9 (`node -v`, `npm -v`).
- **CLI Tools:** `git`, preferred code editor, optional `yarn` or `pnpm`.
- **Local Prep:** Choose a project name (e.g., `my-portfolio`) and decide whether the site will live at `<username>.github.io` or under a subpath.

---

## 3. Bootstrap the Docusaurus Project

### 3.1 Create the Workspace

```bash
npm init docusaurus@latest my-portfolio classic
cd my-portfolio
npm install
```

- The **classic template** ships with docs, blog, and custom pages; adjust answers in the scaffold prompts as needed.

### 3.2 Run the Development Server

```bash
npm run start
```

- Opens `http://localhost:3000` with hot reload. Keep it running while editing.

### 3.3 Project Layout Overview

```
my-portfolio/
├── blog/                  # Blog posts in Markdown/MDX
├── docs/                  # Documentation/portfolio pages
├── src/pages/             # Standalone React/MDX pages (e.g., /about)
├── static/                # Assets copied verbatim to the build
├── docusaurus.config.js   # Global site configuration
└── sidebars.js            # Sidebar structure for docs
```

---

## 4. Customize Content

### 4.1 Docs Section

- Create folders in `docs/` for resume highlights, publications, or tutorials.
- Update `sidebars.js` to group sections (e.g., `Career`, `Projects`, `Talks`).
- Support MDX: embed JSX components for callouts, badges, or charts.

### 4.2 Blog

- Remove sample posts and create new ones using `npm run new blog`.
- Add front matter with `title`, `authors`, and `tags` to improve navigation.

### 4.3 Landing and About Pages

- Edit `src/pages/index.js` to tailor the hero banner, call-to-action buttons, and feature cards.
- Add an `src/pages/about.mdx` for a bio or CV summary.
- Use components from `@docusaurus/theme-classic` for consistent styling.

### 4.4 Assets and Metadata

- Place profile images, logos, or downloadable resumes under `static/`.
- Update `static/img/favicon.ico` and `static/img/logo.svg` for branding consistency.

---

## 5. Configure Site Identity

Modify `docusaurus.config.js`:

- `title`, `tagline`, `favicon`, `url`, and `baseUrl` (see Section 6).
- `organizationName` and `projectName` must match GitHub repo names.
- `themeConfig.navbar` for top navigation links to docs, blog, GitHub, LinkedIn, etc.
- `themeConfig.footer` for contact info, social links, and copyright.
- Add `metadata` entries (keywords, description) to improve SEO.

Optional enhancements:

- Integrate Google Analytics, Plausible, or Giscus comments via plugins.
- Enable Prism themes for code syntax highlighting.

---

## 6. Prepare for GitHub Pages Deployment

### 6.1 Decide the Publishing Path

- **User/Org site:** Repository named `<username>.github.io` with `baseUrl: "/"`.
- **Project site:** Any other repo name; set `baseUrl: "/<repo>/"`.

### 6.2 Update Configuration

In `docusaurus.config.js`:

```js
const config = {
  url: 'https://<username>.github.io',
  baseUrl: process.env.DEPLOY_BASE_URL ?? '/',
  organizationName: '<username>',
  projectName: 'my-portfolio',
  trailingSlash: false,
  deploymentBranch: 'gh-pages',
  // ...
};
```

- Use environment variables (`DEPLOY_BASE_URL`) for flexible staging builds.

### 6.3 Build Locally to Verify

```bash
npm run build
npm run serve    # Optional: preview the production bundle locally
```

- Fix broken links reported during the build before publishing.

---

## 7. Publish to GitHub Pages

### 7.1 Initialize Git

```bash
git init
git add .
git commit -m "feat: bootstrap personal website"
```

### 7.2 Create the Remote Repository

```bash
gh repo create <username>/my-portfolio --public --source=. --remote=origin
git push -u origin main
```

*(Use `gh` CLI or create the repo manually on github.com.)*

### 7.3 Configure GitHub Actions Deployment

Add `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 18
          cache: npm
      - run: npm ci
      - run: npm run build
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build
          publish_branch: gh-pages
```

- In the repository settings, enable GitHub Pages for branch `gh-pages`.
- First push triggers the workflow; monitor progress in **Actions**.

### 7.4 Manual One-Off Deployment (Optional)

```bash
npx docusaurus deploy
```

- Uses the `deploymentBranch` configured earlier. Handy for quick tests before automation.

---

## 8. Enhance the Site

- **Search:** Add Algolia DocSearch or local search plugins.
- **Internationalization:** Configure `i18n` for multilingual content.
- **Custom Components:** Extend `src/theme/` to override or wrap core UI.
- **Callouts & Cards:** Use MDX components (`admonitions`, `Tabs`, `CodeBlock`) to present highlights cleanly.
- **Performance:** Optimize images with `sharp` or `ImageOptim` before placing them in `static/`.

---

## 9. Maintenance Workflow

- Create feature branches for significant edits; open pull requests for review.
- Use `npm run lint` or add `eslint` + `prettier` to enforce formatting.
- Snapshot major milestones by tagging releases (`git tag v1.0.0`).
- Periodically upgrade Docusaurus with `npm outdated` and review release notes.

---

## 10. Troubleshooting

- **Broken baseUrl:** Verify `config.baseUrl` matches the deployed path; incorrect values break CSS and navigation.
- **Missing assets:** Files must live under `static/`; rename with lowercase, hyphenated paths.
- **404 on refresh:** Ensure GitHub Pages serves from `gh-pages` and SPA fallback is enabled automatically by Docusaurus.
- **Action fails on build:** Clear caches: `npm cache clean --force` locally; rerun with `npm ci`.
- **HTTPS issues:** Custom domains require DNS `CNAME` pointing to GitHub; add the domain to `static/CNAME`.

---

## 11. Recommended Directory Structure for Production

```
my-portfolio/
├── docs/
│   ├── career/
│   ├── projects/
│   └── publications/
├── blog/
├── src/
│   ├── components/      # Reusable React/MDX building blocks
│   └── pages/
├── static/
│   ├── img/
│   └── CNAME            # Optional custom domain
├── .github/workflows/
│   └── deploy.yml
├── docusaurus.config.js
├── package.json
└── README.md
```

---

*Author: Liangchao Deng, Ph.D. Candidate, Shihezi University / CAS-CEMPS*  
*Tutorial prepared for personal portfolio deployment workshops.*
