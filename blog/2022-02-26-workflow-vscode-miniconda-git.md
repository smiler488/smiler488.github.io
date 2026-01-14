---
title: "A Complete Workflow Guide: Using VS Code, Miniconda, and Git for Research Projects"
slug: workflow-vscode-miniconda-git
description: Streamlined setup and collaboration playbook combining VS Code, Miniconda, and Git for reproducible research projects.
authors: [liangchao]
tags: [Python, Git, VSCode, Miniconda, tutorial]
image: /img/blog-default.jpg
---

## Project Overview

This playbook outlines a reproducible workflow for Python-focused research—ideal for data analysis, remote sensing, crop modeling, and computational plant science. Follow the sections in order or jump to the one you need.

<!-- truncate -->

# A Complete Workflow Guide: Using VS Code, Miniconda, and Git for Research Projects

---

## 1. Environment Setup

### 1.1 Install Visual Studio Code

- Download the installer from [code.visualstudio.com](https://code.visualstudio.com/).
- Recommended extensions: Python (Microsoft), Jupyter, GitLens, Pylance, Markdown All in One, Remote - SSH.

### 1.2 Install Miniconda

- Download from [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
- Verify the installation:

```bash
conda --version
```

- Keep Conda updated:

```bash
conda update conda
```

### 1.3 Install Git

- Download from [git-scm.com/downloads](https://git-scm.com/downloads).
- Confirm the version and configure your identity:

```bash
git --version
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 2. Project Initialization and Environment Management

### 2.1 Create the Project Folder

```bash
mkdir cotton_modeling
cd cotton_modeling
```

### 2.2 Initialize Git

```bash
git init
```

### 2.3 Create the Conda Environment

```bash
conda create -n cotton python=3.10
conda activate cotton
```

### 2.4 Install Core Packages

```bash
conda install numpy pandas matplotlib scikit-learn
conda install -c conda-forge opencv open3d
```

### 2.5 Share the Environment

```bash
conda env export > environment.yml
```

To reproduce the environment elsewhere:

```bash
conda env create -f environment.yml
```

---

## 3. Set Up the Project in VS Code

### 3.1 Open the Workspace

- Launch VS Code → `File > Open Folder…` → choose the project directory.
- Select the Python interpreter (`Command Palette > Python: Select Interpreter`) and pick `conda: cotton`.

### 3.2 Suggested Project Structure

```
cotton_modeling/
├── data/              # Raw & processed datasets (not committed)
├── notebooks/         # Jupyter notebooks
├── scripts/           # Core Python modules
│   ├── preprocessing.py
│   ├── modeling.py
│   └── visualization.py
├── results/           # Generated figures, tables, reports
├── environment.yml    # Conda spec for reproducibility
├── README.md          # Project overview and usage
└── .gitignore
```

### 3.3 `.gitignore` Essentials

```
__pycache__/
*.pyc
*.ipynb_checkpoints
data/
results/
.env
```

---

## 4. Git Workflow (Single Researcher)

### 4.1 Stage and Commit Changes

```bash
git add .
git commit -m "Initial commit: data preprocessing pipeline"
```

### 4.2 Connect to GitHub

```bash
git remote add origin https://github.com/yourname/cotton_modeling.git
git branch -M main
git push -u origin main
```

### 4.3 Sync Regularly

```bash
git pull origin main
git push origin main
```

### 4.4 Use Feature Branches

```bash
git checkout -b feature-light-simulation
# Implement changes...
git add .
git commit -m "Add light simulation module"
git push origin feature-light-simulation
```

---

## 5. Collaboration Workflow

1. Fork the repository and clone locally:

   ```bash
   git clone https://github.com/leader/cotton_modeling.git
   ```

2. Create a feature branch:

   ```bash
   git checkout -b analysis-update
   ```

3. Commit and push updates:

   ```bash
   git add .
   git commit -m "Update canopy reflectance model"
   git push origin analysis-update
   ```

4. Open a pull request on GitHub for review and merging.

---

## 6. Maintenance and Reproducibility

- **Keep environments current:** `conda env export > environment.yml`
- **Document clearly:** Maintain `README.md` with project overview, requirements, usage, and data notes; use docstrings for modules.
- **Tag releases:** `git tag -a v1.0 -m "First release"` then `git push origin v1.0`.
- **Manage data responsibly:** Keep raw data read-only, avoid committing large binaries, update `.gitignore` to exclude generated files.

---

## 7. Typical Research Project Flow

1. Initialize the repository with Git.
2. Create and activate the Conda environment.
3. Develop scripts and notebooks in VS Code.
4. Commit frequently and push to GitHub.
5. Branch for experiments or new modules.
6. Export results and environment descriptors.
7. Reference commit hashes or tags in publications for transparency.

---

## 8. Common Issues and Fixes

| Issue | Quick Fix |
| --- | --- |
| VS Code cannot find the Conda environment | Use `Python: Select Interpreter` and choose the correct Conda env. |
| `git push` authentication errors | Refresh your GitHub token or sign in again using `gh auth login`. |
| Conda dependency conflicts | Run `conda clean --all` or recreate the environment from `environment.yml`. |
| Jupyter kernel missing | Install kernel: `python -m ipykernel install --user --name=cotton`. |

---

## 9. Final Notes

Adopting VS Code, Miniconda, and Git as a unified workflow delivers:
- Reproducibility: every environment and code change is versioned.
- Transparency: collaboration and provenance are traceable.
- Efficiency: tooling accelerates experimentation and debugging.

---

*Author: Liangchao Deng*  
*Website: smiler488.github.io*
