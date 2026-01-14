---
slug: gitHub-beginner-guide
title: GitHub Beginner Guide
authors: [liangchao]
tags: [tutorial]
image: /img/blog-default.jpg
---


## Project Overview
GitHub is a web-based platform for version control and collaboration. It allows multiple people to work on projects together, track changes, and manage code repositories using Git.

<!-- truncate -->

## 1. Setting Up Git and GitHub

### **1.1 Create a GitHub Account**
1. Go to [GitHub](https://github.com/).
2. Click **Sign up** and fill in your details.
3. Verify your email and set up your profile.

### **1.2 Install Git**
#### **Windows:**
Download and install Git from [Git for Windows](https://git-scm.com/).

#### **Mac:**
```bash
brew install git
```

#### **Linux:**
```bash
sudo apt update
sudo apt install git
```

### **1.3 Configure Git**
Set up your Git username and email:
```bash
git config --global user.name "Your GitHub Username"
git config --global user.email "Your GitHub Email"
```
Check the configuration:
```bash
git config --list
```

---

## 2. Creating a Repository on GitHub
1. Log in to [GitHub](https://github.com/).
2. Click **New repository**.
3. Enter a **repository name**, select visibility (Public or Private), and click **Create repository**.

---

## 3. Cloning a Repository
To copy a GitHub repository to your local machine:
```bash
git clone https://github.com/your-username/repository-name.git
```

Move into the directory:
```bash
cd repository-name
```

---

## 4. Adding and Committing Changes
### **4.1 Create a new file**
```bash
echo "# My Project" > README.md
```

### **4.2 Add files to staging**
```bash
git add README.md
```

### **4.3 Commit changes**
```bash
git commit -m "Initial commit"
```

---

## 5. Pushing Code to GitHub
```bash
git push origin main
```

If your branch is different from `main`, use:
```bash
git push origin your-branch-name
```

---

## 6. Pulling Updates from GitHub
To get the latest changes from GitHub:
```bash
git pull origin main
```

---

## 7. Branching and Merging
### **7.1 Create a New Branch**
```bash
git branch feature-branch
```

### **7.2 Switch to the New Branch**
```bash
git checkout feature-branch
```

### **7.3 Merge a Branch**
```bash
git checkout main
git merge feature-branch
```

### **7.4 Delete a Branch**
```bash
git branch -d feature-branch
```

---

## 8. Forking a Repository and Creating Pull Requests
### **8.1 Fork a Repository**
1. Go to the repository on GitHub.
2. Click **Fork** (top-right corner).
3. Clone your forked repository:
```bash
git clone https://github.com/your-username/forked-repository.git
```

### **8.2 Make Changes and Push**
```bash
git add .
git commit -m "Modified file"
git push origin your-branch
```

### **8.3 Create a Pull Request (PR)**
1. Go to your forked repository on GitHub.
2. Click **New pull request**.
3. Compare changes and click **Create pull request**.

---

## 9. Git Ignore and Undo Changes
### **9.1 Ignoring Files**
Create a `.gitignore` file and add files or folders you want to ignore:
```
node_modules/
*.log
.env
```

### **9.2 Undo Changes**
#### **Undo uncommitted changes:**
```bash
git checkout -- filename
```

#### **Undo last commit (keep changes unstaged):**
```bash
git reset --soft HEAD~1
```

#### **Undo last commit (discard changes):**
```bash
git reset --hard HEAD~1
```

---

## 10. Useful Git Commands Summary
| Command | Description |
|---------|-------------|
| `git init` | Initialize a Git repository |
| `git clone URL` | Clone a repository |
| `git status` | Show current changes |
| `git add .` | Add all files to staging |
| `git commit -m "message"` | Commit changes |
| `git push origin branch` | Push changes to GitHub |
| `git pull origin branch` | Pull changes from GitHub |
| `git branch branch-name` | Create a new branch |
| `git checkout branch-name` | Switch branches |
| `git merge branch-name` | Merge branches |
| `git reset --hard HEAD~1` | Undo last commit |
| `git log` | View commit history |

---

## Conclusion
This guide covers the basics of Git and GitHub. As you become more familiar, you can explore advanced topics such as GitHub Actions, contributing to open-source projects, and automated deployments.
