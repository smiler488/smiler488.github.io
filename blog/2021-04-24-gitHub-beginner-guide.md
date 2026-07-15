---
slug: gitHub-beginner-guide
title: Git and GitHub Beginner Guide
description: A safe, modern introduction to repositories, commits, branches, remotes, pull requests, authentication, and undoing mistakes.
authors: [liangchao]
category: Developer tools
article_type: Technical guide
tags: [git, reproducible-research, web-development]
image: /img/blog-default.jpg
---

## Project overview

Git records changes to files on your computer. GitHub hosts Git repositories and adds collaboration features such as pull requests, issues, and automated checks. This guide follows a complete first workflow and separates safe recovery commands from history-rewriting operations.

<!-- truncate -->

## 1. Install and identify yourself

### Install Git

- **Windows:** download [Git for Windows](https://git-scm.com/).
- **macOS:** install the Xcode Command Line Tools or use Homebrew:

  ```bash
  brew install git
  ```

- **Ubuntu or Debian:**

  ```bash
  sudo apt update
  sudo apt install git
  ```

Confirm the installation:

```bash
git --version
```

### Configure commit identity

Use the name you want shown in commit history. The email should be an address associated with your GitHub account, or your GitHub-provided private `noreply` address.

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

Review the configuration:

```bash
git config --global --list
```

## 2. Create a repository

Create a folder locally:

```bash
mkdir my-project
cd my-project
git init
```

Add a minimal project description:

```bash
echo "# My Project" > README.md
git status
git add README.md
git commit -m "docs: add project overview"
```

Before staging data, credentials, or generated files, create a `.gitignore`:

```text
.env
node_modules/
__pycache__/
*.log
```

Never commit API keys or passwords. Removing a secret in a later commit does not remove it from earlier history; rotate an exposed credential immediately.

## 3. Connect the GitHub remote

Create an empty repository on [GitHub](https://github.com/) without initializing another README, then connect it:

```bash
git branch -M main
git remote add origin https://github.com/your-username/repository-name.git
git remote -v
git push -u origin main
```

GitHub does not accept an account password for Git operations over HTTPS. Use a supported credential manager, GitHub CLI, a personal access token, or SSH authentication.

For an existing repository:

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

## 4. The everyday edit cycle

Inspect changes before staging:

```bash
git status
git diff
```

Stage intentionally and commit a coherent unit:

```bash
git add path/to/file
git diff --staged
git commit -m "feat: describe the change"
git push
```

Prefer specific paths over `git add .` when a workspace contains unrelated or generated changes.

Before starting new work on a shared branch:

```bash
git fetch origin
git status
git pull --ff-only origin main
```

`--ff-only` refuses to create an unexpected merge commit. If local and remote histories have diverged, inspect them and decide whether to rebase or merge rather than forcing the operation.

## 5. Work on a branch

Create and switch to a feature branch:

```bash
git switch -c feature-clear-name
```

Commit and publish it:

```bash
git add path/to/file
git commit -m "feat: add clear capability"
git push -u origin feature-clear-name
```

After review, return to the default branch and update it:

```bash
git switch main
git pull --ff-only origin main
```

Delete a fully merged local branch:

```bash
git branch -d feature-clear-name
```

## 6. Forks and pull requests

Use a fork when you do not have permission to push branches to the upstream repository.

1. Open the upstream repository and select **Fork**.
2. Clone your fork, not the original repository:

   ```bash
   git clone https://github.com/your-username/forked-repository.git
   cd forked-repository
   ```

3. Add the original repository as `upstream`:

   ```bash
   git remote add upstream https://github.com/original-owner/repository.git
   git fetch upstream
   ```

4. Create a branch, commit, and push it to your fork:

   ```bash
   git switch -c analysis-update
   git add path/to/file
   git commit -m "feat: update analysis"
   git push -u origin analysis-update
   ```

5. On GitHub, open a pull request from the fork branch to the upstream default branch.

To synchronize later:

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

## 7. Undo changes safely

Choose the command based on where the mistake exists.

### Discard an unstaged file edit

```bash
git restore path/to/file
```

This permanently replaces the working copy with the last committed version.

### Unstage a file but keep its edits

```bash
git restore --staged path/to/file
```

### Correct the most recent local commit

If it has not been pushed:

```bash
git add path/to/fix
git commit --amend
```

### Reverse a published commit

Create a new commit that reverses the selected commit:

```bash
git log --oneline
git revert <commit-hash>
```

:::warning Avoid destructive history changes on shared branches
`git reset --hard` discards local work, and force-pushing rewrites shared history. They are not routine beginner recovery tools. Make a backup branch and confirm the collaboration policy before using either.
:::

## 8. Useful inspection commands

| Command                                      | Purpose                            |
| -------------------------------------------- | ---------------------------------- |
| `git status`                                 | Show branch and working-tree state |
| `git diff`                                   | Show unstaged changes              |
| `git diff --staged`                          | Show staged changes                |
| `git log --oneline --graph --decorate --all` | Inspect branch history             |
| `git remote -v`                              | Show remote names and URLs         |
| `git branch -vv`                             | Show branches and their upstreams  |
| `git show <commit>`                          | Inspect one commit                 |

## Final checklist

Before pushing:

- no secrets or private data are staged;
- generated files are excluded unless intentionally versioned;
- the diff matches the commit message;
- tests or builds relevant to the change pass;
- the destination branch and remote are correct.

For more detail, use the official [Git documentation](https://git-scm.com/doc) and [GitHub Docs](https://docs.github.com/).
