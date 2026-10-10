---
title: 用 VS Code、Miniconda 和 Git 构建可复现的科研工作流
slug: workflow-vscode-miniconda-git
description: "用 VS Code、Conda 环境和 Git 组织 Python 科研项目：记录环境、管理数据、协作开发，保证结果可复现。"
authors: [liangchao]
category: 科研实践
article_type: 工作流
tags: [reproducible-research, python, git, data-analysis]
image: /img/blog-default.jpg
---

## 概述

这套流程把 VS Code、每个项目独立的 Conda 环境和 Git 组合起来。工具只是基础，真正保证可复现的是围绕它们留下的记录：环境说明、只读的原始数据、配置文件、校验值、清晰的提交历史，以及对输出结果的说明。

<!-- truncate -->

## 1. 安装基础工具

- [Visual Studio Code](https://code.visualstudio.com/)，并安装 Python、Pylance 和 Jupyter 扩展
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Git](https://git-scm.com/downloads)

确认终端能找到这些命令：

```bash
code --version
conda --version
git --version
```

Git 只需配置一次：

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

## 2. 创建项目和环境

```bash
mkdir cotton-modeling
cd cotton-modeling
git init

conda create --name cotton python=3.11
conda activate cotton
conda install numpy pandas matplotlib scikit-learn
conda install --channel conda-forge opencv open3d
```

Python 版本和依赖包请按项目实际需要选择，不必照抄上面的例子。

用 VS Code 打开项目文件夹：

```bash
code .
```

在 VS Code 中执行 **Python: Select Interpreter**，选择 `cotton` 环境。

## 3. 适合科研项目的目录结构

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

原始数据只读不改。处理脚本应生成新的结果文件，而不是覆盖输入数据。

## 4. 哪些文件纳入 Git

一个基础的 `.gitignore`：

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

说明文档 `README.md`、数据格式定义、小型测试数据、配置文件，以及复现结果所需的代码，都应纳入版本管理。

数据太大或受使用限制、不适合放进 Git 时：

- 存放在单位认可的数据仓库或对象存储中；
- 记录永久标识符或获取地址；
- 记录校验值和获取日期；
- 只有在 DVC 或 Git LFS 符合团队协作和数据保存要求时才使用它们。

`.env` 中的密钥等凭据绝对不能提交。

## 5. 记录环境

导出显式安装过的包，得到一份跨平台的环境说明：

```bash
conda env export --from-history > environment.yml
```

在另一台机器上重建环境：

```bash
conda env create --file environment.yml
```

`--from-history` 导出的文件简洁、跨平台，但不会锁定所有间接依赖的版本。如果需要精确到具体构建版本，请另外导出平台相关的显式清单或使用锁文件工具，并与发布版本一起归档。

确认项目仍能正常运行后，再更新环境说明：

```bash
conda env export --from-history > environment.yml
git diff environment.yml
```

## 6. Notebook 与代码模块分工

Notebook 用于探索和展示，稳定下来的处理步骤移到 `src/` 中：

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

Notebook 中导入这些函数使用，不要让某项分析只存在于 notebook 里。

为环境注册一个有名字的 Jupyter 内核：

```bash
conda install ipykernel
python -m ipykernel install --user --name cotton --display-name "Python (cotton)"
```

## 7. 提交一组可复现的改动

添加前先检查：

```bash
git status
git diff
```

代码、配置、测试和文档一起提交：

```bash
git add README.md environment.yml configs src tests
git diff --staged
git commit -m "feat: add canopy preprocessing pipeline"
```

关联 GitHub 仓库：

```bash
git branch -M main
git remote add origin https://github.com/yourname/cotton_modeling.git
git push -u origin main
```

开始新工作前先同步：

```bash
git fetch origin
git status
git pull --ff-only origin main
```

## 8. 选择协作方式

### 共用一个仓库

有写权限的成员在同一个仓库中各自建分支：

```bash
git switch -c feature-light-simulation
# 编辑并测试
git add configs src tests
git commit -m "feat: add light simulation module"
git push -u origin feature-light-simulation
```

发起拉取请求，审阅改动和自动检查结果后再合并。

### 通过 fork 贡献

没有写权限的成员克隆自己的 fork，并把原仓库添加为 `upstream`：

```bash
git clone https://github.com/yourname/cotton_modeling.git
cd cotton_modeling
git remote add upstream https://github.com/leader/cotton_modeling.git
git fetch upstream
git switch -c analysis-update
```

把分支推送到自己的 fork，再向上游的 `main` 发起拉取请求。

## 9. 可复现性清单

每次分析或模型运行，都应保存：

- 代码对应的提交；正式发布的版本另打带说明的 Git 标签；
- 环境说明或锁文件；
- 输入数据的标识符、版本、许可和校验值；
- 配置参数和随机种子；
- 结果受硬件或加速器影响时，相应的硬件信息；
- 运行流程所用的命令；
- 生成的日志、指标，以及对预期输出的说明；
- 尚未自动化的手动步骤。

除非实际测试过，不要声称跨平台逐位一致。更实际、也更有意义的目标是：按照记录的流程，能在规定的误差范围内复现科学结论。

## 常见问题

| 问题 | 处理方法 |
| --- | --- |
| VS Code 用错了 Python | 执行 **Python: Select Interpreter**，再用 `python -c "import sys; print(sys.executable)"` 确认。 |
| Notebook 找不到内核 | 在环境中安装 `ipykernel` 并注册内核。 |
| Conda 无法解析依赖 | 去掉不必要的版本限制，新建环境，并记录最终解析出的版本。 |
| 推送时认证失败 | 使用 GitHub CLI、凭据管理器、个人访问令牌或 SSH；Git 推送不接受账号密码。 |
| 误提交了大数据集 | 先停下，确认这段历史是否已推送，再按仓库规定的流程删除数据。 |
