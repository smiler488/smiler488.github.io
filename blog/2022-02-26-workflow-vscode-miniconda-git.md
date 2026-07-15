---
title: A Reproducible Research Workflow with VS Code, Miniconda, and Git
slug: workflow-vscode-miniconda-git
description: A practical workflow for structuring Python research projects, recording environments, collaborating through Git, and preserving data provenance.
authors: [liangchao]
category: Research practice
article_type: Workflow
tags: [reproducible-research, python, git, data-analysis]
image: /img/blog-default.jpg
---

## Project overview

This workflow combines VS Code, a project-specific Conda environment, and Git. The tools are useful, but reproducibility comes from the records around them: environment specifications, immutable raw data, configuration files, checksums, clear commits, and documented outputs.

<!-- truncate -->

## 1. Install the core tools

- [Visual Studio Code](https://code.visualstudio.com/) with the Python, Pylance, and Jupyter extensions
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Git](https://git-scm.com/downloads)

Confirm that the active shell can find them:

```bash
code --version
conda --version
git --version
```

Configure Git once:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

## 2. Create a project and environment

```bash
mkdir cotton-modeling
cd cotton-modeling
git init

conda create --name cotton python=3.11
conda activate cotton
conda install numpy pandas matplotlib scikit-learn
conda install --channel conda-forge opencv open3d
```

Choose the Python version and packages required by the project rather than copying this example unchanged.

Open the folder:

```bash
code .
```

In VS Code, run **Python: Select Interpreter** and select the `cotton` environment.

## 3. Use a research-friendly structure

```text
cotton-modeling/
├── README.md
├── environment.yml
├── pyproject.toml          # optional package and tool configuration
├── configs/                # versioned experiment parameters
├── data/
│   ├── README.md           # source, license, schema, and retrieval notes
│   ├── raw/                # immutable source data
│   └── processed/          # reproducible derived data
├── notebooks/              # exploration, not the only implementation
├── src/cotton_modeling/    # reusable Python modules
├── tests/
├── results/
│   └── README.md           # explains how outputs are generated
└── .gitignore
```

Keep raw data immutable. A processing script should create a new derived artifact rather than overwrite its input.

## 4. Decide what Git should track

A starting `.gitignore`:

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

Do not ignore the explanatory `README.md` files, schemas, small test fixtures, configuration, or the code required to reproduce an output.

For data too large or restricted for Git:

- store it in an approved repository or object store;
- record a persistent identifier or retrieval location;
- record checksums and access dates;
- use DVC or Git LFS only when they fit the collaboration and preservation plan.

Never commit credentials from `.env`.

## 5. Record the environment

For a portable list of the packages you explicitly requested:

```bash
conda env export --from-history > environment.yml
```

Recreate it with:

```bash
conda env create --file environment.yml
```

`--from-history` is readable and cross-platform, but it does not lock every transitive dependency. When exact package builds matter, create a platform-specific explicit specification or use a lock-file tool, then archive that file with the release.

Update the environment description only after confirming the project still runs:

```bash
conda env export --from-history > environment.yml
git diff environment.yml
```

## 6. Keep notebooks and modules in sync

Use notebooks for exploration and communication, but move stable operations into `src/`:

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

Notebooks should import these functions instead of containing the only copy of an analysis.

For a named Jupyter kernel:

```bash
conda install ipykernel
python -m ipykernel install --user --name cotton --display-name "Python (cotton)"
```

## 7. Commit a reproducible unit of work

Review before staging:

```bash
git status
git diff
```

Commit code, configuration, tests, and documentation together:

```bash
git add README.md environment.yml configs src tests
git diff --staged
git commit -m "feat: add canopy preprocessing pipeline"
```

Connect a GitHub repository:

```bash
git branch -M main
git remote add origin https://github.com/yourname/cotton_modeling.git
git push -u origin main
```

Before starting new work:

```bash
git fetch origin
git status
git pull --ff-only origin main
```

## 8. Choose one collaboration model

### Shared repository

Members with write access create branches in the same repository:

```bash
git switch -c feature-light-simulation
# edit and test
git add configs src tests
git commit -m "feat: add light simulation module"
git push -u origin feature-light-simulation
```

Open a pull request, review the diff and checks, then merge.

### Fork-based contribution

Contributors without write access clone their own fork and register the original repository as `upstream`:

```bash
git clone https://github.com/yourname/cotton_modeling.git
cd cotton_modeling
git remote add upstream https://github.com/leader/cotton_modeling.git
git fetch upstream
git switch -c analysis-update
```

Push the branch to the fork and open a pull request against `upstream/main`.

## 9. Reproducibility checklist

For each analysis or model run, preserve:

- code commit and, for releases, an annotated Git tag;
- environment or lock file;
- input dataset identifier, version, license, and checksum;
- configuration and random seeds;
- hardware or accelerator details when results are sensitive to them;
- commands used to run the workflow;
- generated logs, metrics, and a description of expected outputs;
- any manual step that cannot yet be automated.

Do not claim bit-for-bit reproducibility across platforms unless it has been tested. The stronger and usually more useful target is a documented workflow that reproduces the scientific conclusion within defined tolerances.

## Common issues

| Issue                           | Check                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| VS Code uses the wrong Python   | Run **Python: Select Interpreter** and verify `python -c "import sys; print(sys.executable)"`.         |
| Notebook kernel is missing      | Install `ipykernel` inside the environment and register the kernel.                                    |
| Conda cannot solve dependencies | Remove unnecessary constraints, create a fresh environment, and document the resolved set.             |
| Push authentication fails       | Use GitHub CLI, a credential manager, PAT, or SSH; account passwords are not accepted for Git pushes.  |
| A large dataset was committed   | Stop, review whether history was published, and follow the repository's approved data-removal process. |
