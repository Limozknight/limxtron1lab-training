# 创建你自己的 GitHub 仓库 - 完整步骤

## 🚀 第 1 步：在 GitHub 上创建新仓库

1. 打开 GitHub 网站：https://github.com
2. 登录你的账号
3. 点击右上角的 **➕ 号** → **New repository**

### 填写仓库信息：

```
Repository name:     limxtron1lab-training
                     ↑ 或任何你喜欢的名字

Description:         LIMX TRON1A Bipedal Robot RL Training
                     (使用Isaac Lab + PPO)

Public / Private:    选择 Public（如果要在网站使用）
                     或 Private（个人项目）

Initialize:          ❌ 不要勾选任何选项
                     因为本地已有代码
```

4. 点击 **Create repository** 按钮

### 创建完成后，你会看到类似这样的页面：

```
Quick setup — if you've done this kind of thing before

or
https://github.com/YOUR_USERNAME/limxtron1lab-training.git

...or push an existing repository from the command line

git remote add origin https://github.com/YOUR_USERNAME/limxtron1lab-training.git
git branch -M main
git push -u origin main
```

**记住你的仓库 URL**（下一步需要）

---

## 🔗 第 2 步：配置本地 Git（关键！）

打开 PowerShell，执行：

```bash
# 进入项目目录
cd "c:\Users\17950\Desktop\everything\IE\SDM5008\limxtron1lab-main (1)\limxtron1lab-main"

# 替换 YOUR_USERNAME 和 REPO_NAME 为实际的用户名和仓库名
git remote add origin https://github.com/YOUR_USERNAME/limxtron1lab-training.git

# 验证配置成功
git remote -v
```

应该看到：
```
origin  https://github.com/YOUR_USERNAME/limxtron1lab-training.git (fetch)
origin  https://github.com/YOUR_USERNAME/limxtron1lab-training.git (push)
```

---

## 📤 第 3 步：推送代码到你的仓库

### 选项 A：使用 master 分支（推荐）

```bash
git push -u origin master
```

### 选项 B：重命名为 main 后推送（符合现代标准）

```bash
# 重命名分支
git branch -m master main

# 推送
git push -u origin main
```

---

## ✅ 验证成功

推送完成后，打开 GitHub 仓库链接：
```
https://github.com/YOUR_USERNAME/limxtron1lab-training
```

应该能看到：
- ✅ 所有的代码文件（exts/, rsl_rl/, scripts/ 等）
- ✅ 所有的文档（docs/ 目录）
- ✅ README.md
- ✅ 提交历史

---

## 🎯 快速参考命令

### 假设你的用户名是 `UserName`，仓库名是 `limxtron1lab-training`

```bash
# 进入项目
cd "c:\Users\17950\Desktop\everything\IE\SDM5008\limxtron1lab-main (1)\limxtron1lab-main"

# 第 1 次推送（配置远程）
git remote add origin https://github.com/UserName/limxtron1lab-training.git
git push -u origin master

# 之后的推送（简单）
git add .
git commit -m "Update: ..."
git push
```

---

## 🔑 如果失败了？

### 错误 1: `fatal: unable to access 'https://github.com/...'`

**原因**: GitHub 不再支持密码认证

**解决方案**：

#### 方法 1：使用个人访问令牌（推荐）

1. 打开 GitHub Settings → Developer settings → Personal access tokens
2. 点击 "Generate new token"
3. 选择 `repo` 权限范围
4. 复制生成的 token

然后在 PowerShell 中，当提示输入密码时，**粘贴 token**（不是密码）

#### 方法 2：使用 SSH（更好）

```bash
# 1. 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "your@email.com"
# 或者（Windows）
ssh-keygen -t rsa -b 4096 -C "your@email.com"

# 2. 按 Enter 几次接受默认设置

# 3. 显示公钥
type $PROFILE\.ssh\id_ed25519.pub

# 4. 复制输出的公钥

# 5. 在 GitHub Settings → SSH and GPG keys → New SSH key
#    粘贴公钥

# 6. 使用 SSH 配置远程
git remote add origin git@github.com:YOUR_USERNAME/limxtron1lab-training.git
git push -u origin master
```

---

### 错误 2: `error: src refspec master does not match any`

**原因**: 没有任何提交历史

**解决方案**：
```bash
# 确保有提交
git log --oneline

# 如果没有，先提交
git add .
git commit -m "Initial commit"

# 再推送
git push -u origin master
```

---

## 📋 完整步骤总结

```
1️⃣ GitHub 上创建新仓库
   ↓
2️⃣ 复制仓库 URL
   ↓
3️⃣ 本地删除旧远程配置（已做）
   git remote remove origin
   ↓
4️⃣ 添加新远程配置
   git remote add origin <你的仓库URL>
   ↓
5️⃣ 验证配置
   git remote -v
   ↓
6️⃣ 推送代码
   git push -u origin master
   ↓
7️⃣ 在浏览器验证
   https://github.com/YOUR_USERNAME/limxtron1lab-training
```

---

## 💡 提示

**如果你想改项目名字**：
```bash
# 本地重命名文件夹（可选）
# 然后在 GitHub 设置中修改仓库名
```

**如果要添加 .gitignore**：
```bash
# 已经有 .gitignore，但如果需要更新
# 编辑 .gitignore 文件，然后：
git add .gitignore
git commit -m "Update .gitignore"
git push
```

**以后每次更新代码**：
```bash
git add .
git commit -m "描述你的修改"
git push  # 就是这样简单
```

---

