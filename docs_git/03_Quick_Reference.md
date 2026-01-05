# Git 快速参考卡 - 一页纸包含所有你需要知道的

## 🎯 最常用的 5 个命令

```bash
git status              # 看现在什么状态
git add .               # 把所有修改加入暂存区
git commit -m "msg"     # 提交（本地保存）
git push                # 上传到 GitHub
git log --oneline       # 看提交历史
```

---

## 🔄 典型的一天工作流

```
早上：
  git pull                  # 获取最新代码

工作中：
  # ... 修改文件 ...
  git status                # 检查修改
  git diff file.py          # 看具体改了什么

下班前：
  git add .                 # 添加所有修改
  git commit -m "..."       # 提交
  git push                  # 上传
```

---

## 🚨 快速修复

### 问题：推送失败

**第 1 步：确认身份**
```bash
git config --global user.name "Limozknight"
git config --global user.email "1795047190@qq.com"
```

**第 2 步：生成 Token**
- 打开: https://github.com/settings/tokens
- 生成新 token（勾选 repo）
- 复制 token

**第 3 步：保存 Token**
```bash
cmdkey /delete:github.com
cmdkey /add:github.com /user:Limozknight /pass:"token"
```

**第 4 步：推送**
```bash
git push -u origin master
```

---

### 问题：改错了，想恢复

```bash
# 还没 add 的
git checkout file.py

# 已经 add 的
git reset HEAD file.py
git checkout file.py

# 已经 commit 的
git revert HEAD
```

---

### 问题：想看之前改过什么

```bash
git log --oneline                    # 所有提交
git log -5                           # 最近 5 个
git diff HEAD~1                      # 对比前一个提交
git show commit_id                   # 看某个提交的详细内容
```

---

## 📊 Git 三个重要区域

```
Working Directory  →  add  →  Staging Area  →  commit  →  Local Repo  →  push  →  GitHub
(你的文件)                    (准备提交)              (本地历史)                  (远程)
```

---

## 🔑 本地配置 vs 项目配置

```bash
# 全局配置（所有项目）
git config --global user.name "Name"

# 仅当前项目
git config user.name "Name"

# 查看
git config --global --list      # 全局配置
git config --list              # 当前项目配置
```

---

## 🌳 分支基础（了解即可）

```bash
git branch                    # 看当前分支
git branch -a                 # 看所有分支
git checkout -b feature       # 创建新分支
git checkout master           # 切换回 master
git merge feature             # 合并分支
```

**现在推荐的做法**：不用分支，直接在 master 上改

---

## 📤 常见推送场景

### 场景 1：第一次推送到新仓库
```bash
git add .
git commit -m "Initial commit"
git push -u origin master      # -u 很重要，建立追踪关系
```

### 场景 2：普通推送（已经设置过 -u）
```bash
git add .
git commit -m "Update something"
git push
```

### 场景 3：推送到不同分支
```bash
git push origin feature_branch
```

---

## 🔍 查看信息的命令

```bash
git status                     # 当前状态
git log                        # 完整提交历史
git log --oneline              # 简洁版本
git log -5                      # 最近 5 个
git log --graph                # 有分支图的版本
git diff                       # 未 add 的修改
git diff --staged              # 已 add 的修改
git show commit_id             # 某个提交的详细内容
git remote -v                  # 远程仓库信息
```

---

## 🎯 你现在需要做的

```bash
# 第 1 步：更新身份
git config --global user.name "Limozknight"
git config --global user.email "1795047190@qq.com"

# 第 2 步：生成 Token（在 GitHub 网站）
# https://github.com/settings/tokens

# 第 3 步：保存 Token
cmdkey /delete:github.com
cmdkey /add:github.com /user:Limozknight /pass:"YOUR_TOKEN"

# 第 4 步：推送
git push -u origin master
```

---

## 📚 相关文件

- `01_Git_Basics.md` - 详细的 Git 概念和工作流
- `02_Fix_Your_Push_Problem.md` - 详细的推送问题解决方案
- `03_Git_Cheat_Sheet.md` - 完整命令参考

