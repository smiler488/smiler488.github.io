---
slug: gitHub-beginner-guide
title: Git 与 GitHub 入门指南
description: "从安装配置、提交、分支、远程仓库、拉取请求到撤销错误，第一次使用 Git 和 GitHub 需要掌握的完整流程。"
authors: [liangchao]
category: 开发者工具
article_type: 技术指南
tags: [git, reproducible-research, web-development]
image: /img/blog-default.jpg
---

## 概述

Git 用来记录电脑上文件的改动历史；GitHub 托管 Git 仓库，并提供拉取请求（pull request）、issue、自动检查等协作功能。本文带你完整走一遍第一次使用的流程，并把安全的撤销命令和会改写历史的危险操作区分开。

<!-- truncate -->

## 1. 安装与身份设置

### 安装 Git

- **Windows：** 下载 [Git for Windows](https://git-scm.com/)。
- **macOS：** 安装 Xcode 命令行工具，或用 Homebrew 安装：

  ```bash
  brew install git
  ```

- **Ubuntu 或 Debian：**

  ```bash
  sudo apt update
  sudo apt install git
  ```

确认安装成功：

```bash
git --version
```

### 设置提交者信息

用户名就是提交记录里显示的名字。邮箱请填 GitHub 账号绑定的邮箱，或 GitHub 提供的 `noreply` 隐私邮箱。

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

查看当前配置：

```bash
git config --global --list
```

## 2. 创建仓库

在本地新建文件夹并初始化：

```bash
mkdir my-project
cd my-project
git init
```

写一个简短的项目说明并提交：

```bash
echo "# My Project" > README.md
git status
git add README.md
git commit -m "docs: add project overview"
```

在添加数据、密钥或生成文件之前，先创建 `.gitignore`：

```text
.env
node_modules/
__pycache__/
*.log
```

API 密钥和密码绝对不要提交。即使在后面的提交里删掉，它仍然留在之前的历史中；一旦泄露，应立即作废并更换。

## 3. 关联 GitHub 远程仓库

在 [GitHub](https://github.com/) 上新建一个空仓库（不要勾选初始化 README），然后关联：

```bash
git branch -M main
git remote add origin https://github.com/your-username/repository-name.git
git remote -v
git push -u origin main
```

通过 HTTPS 推送时，GitHub 不接受账号密码。请使用凭据管理器、GitHub CLI、个人访问令牌或 SSH 密钥。

克隆已有仓库：

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

## 4. 日常修改流程

添加之前先看改了什么：

```bash
git status
git diff
```

只添加相关的文件，每次提交一组完整的改动：

```bash
git add path/to/file
git diff --staged
git commit -m "feat: describe the change"
git push
```

如果工作区里还有无关改动或生成文件，请写明具体路径，不要用 `git add .`。

在共享分支上开始新工作前，先同步远程：

```bash
git fetch origin
git status
git pull --ff-only origin main
```

`--ff-only` 只允许快进合并，避免意外生成合并提交。如果本地和远程的历史已经分叉，先查看差异，再决定变基还是合并，不要强行覆盖。

## 5. 使用分支

新建并切换到功能分支：

```bash
git switch -c feature-clear-name
```

提交并推送到远程：

```bash
git add path/to/file
git commit -m "feat: add clear capability"
git push -u origin feature-clear-name
```

合并完成后，切回主分支并更新：

```bash
git switch main
git pull --ff-only origin main
```

删除已合并的本地分支：

```bash
git branch -d feature-clear-name
```

## 6. Fork 与拉取请求

没有上游仓库的推送权限时，用 fork 参与贡献：

1. 打开上游仓库，点击 **Fork**。
2. 克隆你自己的 fork，而不是原仓库：

   ```bash
   git clone https://github.com/your-username/forked-repository.git
   cd forked-repository
   ```

3. 把原仓库添加为 `upstream`：

   ```bash
   git remote add upstream https://github.com/original-owner/repository.git
   git fetch upstream
   ```

4. 新建分支、提交，并推送到你的 fork：

   ```bash
   git switch -c analysis-update
   git add path/to/file
   git commit -m "feat: update analysis"
   git push -u origin analysis-update
   ```

5. 在 GitHub 上，从 fork 的分支向上游的默认分支发起拉取请求。

之后与上游保持同步：

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

## 7. 安全地撤销改动

根据改动所处的阶段选择命令。

### 放弃尚未添加的文件修改

```bash
git restore path/to/file
```

文件会被恢复为上一次提交的版本，当前修改无法找回。

### 取消添加，但保留修改

```bash
git restore --staged path/to/file
```

### 修改最近一次本地提交

仅限尚未推送的提交：

```bash
git add path/to/fix
git commit --amend
```

### 撤销已经推送的提交

新建一个反向提交来抵消它：

```bash
git log --oneline
git revert <commit-hash>
```

:::warning 不要在共享分支上改写历史
`git reset --hard` 会丢弃本地修改，强制推送会改写其他人也在用的历史，都不适合作为初学者的常规撤销手段。确实需要时，先建一个备份分支，并和协作者确认。
:::

## 8. 常用查看命令

| 命令 | 作用 |
| --- | --- |
| `git status` | 查看当前分支和工作区状态 |
| `git diff` | 查看尚未添加的修改 |
| `git diff --staged` | 查看已添加、待提交的修改 |
| `git log --oneline --graph --decorate --all` | 查看各分支的提交历史 |
| `git remote -v` | 查看远程仓库名称和地址 |
| `git branch -vv` | 查看分支及其对应的远程分支 |
| `git show <commit>` | 查看某次提交的内容 |

## 推送前检查

- 没有添加密钥或隐私数据；
- 生成文件已排除（除非有意纳入版本管理）；
- 改动内容与提交信息一致；
- 相关测试或构建已通过；
- 推送的目标分支和远程仓库正确。

更多内容请参阅官方 [Git 文档](https://git-scm.com/doc)和 [GitHub 文档](https://docs.github.com/)。
