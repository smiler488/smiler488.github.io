---
title: 用 VS Code、Miniconda 和 Git 构建可复现的科研工作流
slug: workflow-vscode-miniconda-git
description: 一套实用的工作流：组织 Python 科研项目、记录环境、通过 Git 协作，并保全数据溯源。
authors: [liangchao]
category: 科研实践
article_type: 工作流
tags: [reproducible-research, python, git, data-analysis]
image: /img/blog-default.jpg
---

## 项目概述

这套工作流把 VS Code、项目专属的 Conda 环境和 Git 结合起来。工具本身有用，但可复现性来自围绕它们的记录：环境规格、不可变的原始数据、配置文件、校验和、清晰的提交以及有文档说明的输出。

<!-- truncate -->

## 1. 安装核心工具

- [Visual Studio Code](https://code.visualstudio.com/)，并安装 Python、Pylance 和 Jupyter 扩展
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Git](https://git-scm.com/downloads)

确认当前 shell 能找到它们：

```bash
code --version
conda --version
git --version
```

配置一次 Git：

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

## 2. 创建项目与环境

```bash
mkdir cotton-modeling
cd cotton-modeling
git init

conda create --name cotton python=3.11
conda activate cotton
conda install numpy pandas matplotlib scikit-learn
conda install --channel conda-forge opencv open3d
```

请根据项目所需的 Python 版本和包来选择，而不是原样照抄此示例。

打开文件夹：

```bash
code .
```

在 VS Code 中运行 **Python: Select Interpreter**，并选择 `cotton` 环境。

## 3. 使用便于科研的目录结构

```text
cotton-modeling/
├── README.md
├── environment.yml
├── pyproject.toml          # 可选的包与工具配置
├── configs/                # 版本化的实验参数
├── data/
│   ├── README.md           # 来源、许可、模式和获取说明
│   ├── raw/                # 不可变的源数据
│   └── processed/          # 可复现的派生数据
├── notebooks/              # 探索，不是唯一的实现
├── src/cotton_modeling/    # 可复用的 Python 模块
├── tests/
├── results/
│   └── README.md           # 说明输出如何生成
└── .gitignore
```

保持原始数据不可变。处理脚本应创建新的派生产物，而不是覆盖其输入。

## 4. 决定 Git 应追踪什么

一个起步用的 `.gitignore`：

```text
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.env
.DS_Store
data/raw/*
data/processed/*
results/generated/*
```

不要忽略说明性的 `README.md` 文件、模式定义、小型测试夹具、配置，或复现某个输出所需的代码。

对于太大或受限而不适合 Git 的数据：

- 把它存放在批准的仓库或对象存储中；
- 记录一个持久标识符或获取位置；
- 记录校验和与获取日期；
- 仅当 DVC 或 Git LFS 契合协作与留存方案时才使用它们。

绝不要提交 `.env` 中的凭据。

## 5. 记录环境

要得到你显式请求的包的可移植列表：

```bash
conda env export --from-history > environment.yml
```

重新创建它：

```bash
conda env create --file environment.yml
```

`--from-history` 可读且跨平台，但它不会锁定每一个传递依赖。当确切的包构建版本很重要时，创建一个平台相关的显式规格或使用锁文件工具，并把该文件与发布版本一起归档。

只有在确认项目仍能运行之后，才更新环境描述：

```bash
conda env export --from-history > environment.yml
git diff environment.yml
```

## 6. 让 notebook 与模块保持同步

用 notebook 做探索和展示，但把稳定的操作移到 `src/` 中：

```python
# src/cotton_modeling/preprocessing.py
from pathlib import Path

def list_images(folder: str) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(
        path for path in Path(folder).iterdir()
        if path.suffix.lower() in extensions
    )
```

notebook 应当导入这些函数，而不是把某项分析的唯一副本写在 notebook 里。

创建一个具名的 Jupyter kernel：

```bash
conda install ipykernel
python -m ipykernel install --user --name cotton --display-name "Python (cotton)"
```

## 7. 提交一个可复现的工作单元

暂存前先审查：

```bash
git status
git diff
```

把代码、配置、测试和文档一起提交：

```bash
git add README.md environment.yml configs src tests
git diff --staged
git commit -m "feat: add canopy preprocessing pipeline"
```

连接一个 GitHub 仓库：

```bash
git branch -M main
git remote add origin https://github.com/yourname/cotton_modeling.git
git push -u origin main
```

开始新工作之前：

```bash
git fetch origin
git status
git pull --ff-only origin main
```

## 8. 选择一种协作模式

### 共享仓库

有写权限的成员在同一个仓库中创建分支：

```bash
git switch -c feature-light-simulation
# 编辑并测试
git add configs src tests
git commit -m "feat: add light simulation module"
git push -u origin feature-light-simulation
```

发起拉取请求，审查 diff 和检查，然后合并。

### 基于复刻的贡献

没有写权限的贡献者克隆自己的复刻，并把原始仓库注册为 `upstream`：

```bash
git clone https://github.com/yourname/cotton_modeling.git
cd cotton_modeling
git remote add upstream https://github.com/leader/cotton_modeling.git
git fetch upstream
git switch -c analysis-update
```

把分支推送到复刻，并向上游 `upstream/main` 发起拉取请求。

## 9. 可复现性清单

对每次分析或模型运行，保全：

- 代码提交，对于发布版本还要有一个带标注的 Git 标签；
- 环境或锁文件；
- 输入数据集的标识符、版本、许可和校验和；
- 配置和随机种子；
- 当结果对硬件或加速器敏感时的相关细节；
- 运行工作流所用的命令；
- 生成的日志、指标，以及对预期输出的描述；
- 任何尚不能自动化的手动步骤。

不要声称跨平台逐位可复现，除非已做过测试。更强且通常更有用的目标是一个有文档的工作流，能在定义的容差内复现科学结论。

## 常见问题

| 问题                         | 检查                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| VS Code 用了错误的 Python    | 运行 **Python: Select Interpreter**，并验证 `python -c "import sys; print(sys.executable)"`。            |
| Notebook 内核缺失            | 在环境中安装 `ipykernel` 并注册内核。                                                                   |
| Conda 无法解决依赖            | 移除不必要的约束，创建一个新环境，并记录解析出的集合。                                                  |
| 推送认证失败                  | 使用 GitHub CLI、凭据管理器、PAT 或 SSH；账号密码不被 Git 推送接受。                                    |
| 提交了大型数据集              | 停下来，检查历史是否已发布，并按照仓库批准的数据移除流程处理。                                          |
