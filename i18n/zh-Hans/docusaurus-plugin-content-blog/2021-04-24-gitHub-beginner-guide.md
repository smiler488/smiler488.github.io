---
slug: gitHub-beginner-guide
title: Git 与 GitHub 入门指南
description: 一份安全、现代的入门介绍：涵盖仓库、提交、分支、远程、拉取请求、身份验证和撤销错误。
authors: [liangchao]
category: 开发者工具
article_type: 技术指南
tags: [git, reproducible-research, web-development]
image: /img/blog-default.jpg
---

## 项目概述

Git 记录你计算机上文件的改动。GitHub 托管 Git 仓库，并增加拉取请求、issue 和自动化检查等协作功能。本指南遵循一套完整的首次工作流，并把安全的恢复命令与改写历史的操作区分开来。

<!-- truncate -->

## 1. 安装并设置身份

### 安装 Git

- **Windows：** 下载 [Git for Windows](https://git-scm.com/)。
- **macOS：** 安装 Xcode Command Line Tools，或使用 Homebrew：

  ```bash
  brew install git
  ```

- **Ubuntu 或 Debian：**

  ```bash
  sudo apt update
  sudo apt install git
  ```

确认安装：

```bash
git --version
```

### 配置提交身份

使用你希望显示在提交历史中的名字。邮箱应是与你的 GitHub 账号关联的地址，或 GitHub 提供的私有 `noreply` 地址。

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

查看配置：

```bash
git config --global --list
```

## 2. 创建仓库

在本地创建一个文件夹：

```bash
mkdir my-project
cd my-project
git init
```

添加一段简短的项目说明：

```bash
echo "# My Project" > README.md
git status
git add README.md
git commit -m "docs: add project overview"
```

在暂存数据、凭据或生成文件之前，创建一个 `.gitignore`：

```text
.env
node_modules/
__pycache__/
*.log
```

绝不要提交 API 密钥或密码。在后续提交中删除密钥并不能把它从更早的历史中移除；凭据一旦泄露应立即轮换。

## 3. 连接 GitHub 远程仓库

在 [GitHub](https://github.com/) 上创建一个空仓库，不要再初始化一个 README，然后连接它：

```bash
git branch -M main
git remote add origin https://github.com/your-username/repository-name.git
git remote -v
git push -u origin main
```

GitHub 不接受账号密码用于 HTTPS 上的 Git 操作。请使用受支持的凭据管理器、GitHub CLI、个人访问令牌或 SSH 认证。

对于已有仓库：

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

## 4. 日常编辑循环

暂存前先查看改动：

```bash
git status
git diff
```

有意识地暂存，并提交一个连贯的单元：

```bash
git add path/to/file
git diff --staged
git commit -m "feat: describe the change"
git push
```

当工作区中存在无关或生成的改动时，优先使用具体路径而非 `git add .`。

在共享分支上开始新工作之前：

```bash
git fetch origin
git status
git pull --ff-only origin main
```

`--ff-only` 会拒绝创建意料之外的合并提交。如果本地与远程历史已经分叉，先检查它们，再决定是变基还是合并，而不是强行操作。

## 5. 在分支上工作

创建并切换到功能分支：

```bash
git switch -c feature-clear-name
```

提交并发布它：

```bash
git add path/to/file
git commit -m "feat: add clear capability"
git push -u origin feature-clear-name
```

审查后，回到默认分支并更新它：

```bash
git switch main
git pull --ff-only origin main
```

删除已完全合并的本地分支：

```bash
git branch -d feature-clear-name
```

## 6. 复刻与拉取请求

当你没有权限向上游仓库推送分支时，使用复刻（fork）。

1. 打开上游仓库并选择 **Fork**。
2. 克隆你的复刻，而不是原始仓库：

   ```bash
   git clone https://github.com/your-username/forked-repository.git
   cd forked-repository
   ```

3. 把原始仓库添加为 `upstream`：

   ```bash
   git remote add upstream https://github.com/original-owner/repository.git
   git fetch upstream
   ```

4. 创建分支、提交，并推送到你的复刻：

   ```bash
   git switch -c analysis-update
   git add path/to/file
   git commit -m "feat: update analysis"
   git push -u origin analysis-update
   ```

5. 在 GitHub 上，从复刻分支向上游默认分支发起拉取请求。

之后同步：

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

## 7. 安全地撤销改动

根据错误所在的位置选择命令。

### 丢弃未暂存的文件改动

```bash
git restore path/to/file
```

这会用上次提交的版本永久替换工作副本。

### 取消暂存文件但保留其改动

```bash
git restore --staged path/to/file
```

### 修正最近一次本地提交

如果尚未推送：

```bash
git add path/to/fix
git commit --amend
```

### 撤销已发布的提交

创建一个新提交来反转所选提交：

```bash
git log --oneline
git revert <commit-hash>
```

:::warning 避免在共享分支上做破坏性历史改动
`git reset --hard` 会丢弃本地工作，强制推送会改写共享历史。它们不是常规的初学者恢复工具。使用其中任何一个之前，先创建备份分支并确认协作策略。
:::

## 8. 常用查看命令

| 命令                                         | 用途                       |
| -------------------------------------------- | -------------------------- |
| `git status`                                 | 显示分支和工作区状态       |
| `git diff`                                   | 显示未暂存的改动           |
| `git diff --staged`                          | 显示已暂存的改动           |
| `git log --oneline --graph --decorate --all` | 查看分支历史               |
| `git remote -v`                              | 显示远程名称和 URL         |
| `git branch -vv`                             | 显示分支及其上游           |
| `git show <commit>`                          | 查看某一个提交             |

## 最终清单

推送之前：

- 没有密钥或隐私数据被暂存；
- 生成文件已被排除，除非有意纳入版本控制；
- diff 与提交信息一致；
- 与改动相关的测试或构建通过；
- 目标分支和远程正确。

更多细节请查阅官方 [Git 文档](https://git-scm.com/doc) 和 [GitHub 文档](https://docs.github.com/)。
