# GitHub 权限问题解决方案

## 问题诊断

```
error: Permission to Limozknight/limxtron1lab-training.git denied to knightkk4.
```

**原因**: GitHub 用 `knightkk4` 账号去推送到 `Limozknight` 名下的仓库，没有权限

---

## ✅ 解决方案

### 方案 1：使用 GitHub 个人访问令牌（推荐，Windows 最简单）

#### 步骤 1：生成 Personal Access Token

1. 打开 GitHub：https://github.com/settings/tokens
2. 点击 **"Generate new token"** → **"Generate new token (classic)"**
3. 填写信息：
   - **Token name**: `local-training`
   - **Expiration**: `90 days` （或你想要的期限）
   - **Select scopes**: 勾选 ✅ `repo` （完全访问仓库）
4. 点击 **"Generate token"** 按钮
5. **复制生成的 token** （这很重要！关闭页面就看不到了）

#### 步骤 2：在本地保存 token（Windows Credential Manager）

打开 PowerShell，运行：

```bash
# 使用 Credential Manager 保存凭证
cmdkey /add:github.com /user:Limozknight /pass:"你的token"
```

**例如**（假设你的 token 是 `ghp_xxxxx...`）：
```bash
cmdkey /add:github.com /user:Limozknight /pass:"ghp_xxxxx..."
```

#### 步骤 3：推送代码

```bash
cd "c:\Users\17950\Desktop\everything\IE\SDM5008\limxtron1lab-main (1)\limxtron1lab-main"
git push -u origin master
```

系统应该不会再提示输入密码 ✅

---

### 方案 2：使用 SSH 密钥（更安全，推荐）

#### 步骤 1：生成 SSH 密钥

打开 PowerShell，运行：

```bash
ssh-keygen -t ed25519 -C "your@email.com"
```

按以下方式响应：
```
Enter file in which to save the key: 
# 按 Enter 接受默认位置

Enter passphrase (empty for no passphrase): 
# 按 Enter 跳过，不设置密码

Enter same passphrase again:
# 按 Enter
```

#### 步骤 2：获取公钥

```bash
# 显示公钥内容
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

复制输出的内容（从 `ssh-ed25519` 开头到末尾）

#### 步骤 3：在 GitHub 添加 SSH 密钥

1. 打开 GitHub Settings：https://github.com/settings/keys
2. 点击 **"New SSH key"**
3. 填写：
   - **Title**: `Windows Local Machine`
   - **Key type**: `Authentication Key`
   - **Key**: 粘贴你复制的公钥
4. 点击 **"Add SSH key"**

#### 步骤 4：修改本地 Git 配置

```bash
cd "c:\Users\17950\Desktop\everything\IE\SDM5008\limxtron1lab-main (1)\limxtron1lab-main"

# 将 HTTPS 改为 SSH
git remote set-url origin git@github.com:Limozknight/limxtron1lab-training.git

# 验证
git remote -v
```

应该看到：
```
origin  git@github.com:Limozknight/limxtron1lab-training.git (fetch)
origin  git@github.com:Limozknight/limxtron1lab-training.git (push)
```

#### 步骤 5：推送代码

```bash
git push -u origin master
```

---

## 🚀 快速选择：

**如果你想快速解决**（用方案 1）：
```bash
# 1. 生成 token（在 GitHub）
#    https://github.com/settings/tokens

# 2. 保存 token（在 PowerShell）
cmdkey /add:github.com /user:Limozknight /pass:"你的token"

# 3. 推送
cd "c:\Users\17950\Desktop\everything\IE\SDM5008\limxtron1lab-main (1)\limxtron1lab-main"
git push -u origin master
```

**如果你想更安全**（用方案 2）：
```bash
# 1. 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your@email.com"

# 2. 查看公钥
type $env:USERPROFILE\.ssh\id_ed25519.pub

# 3. 在 GitHub 添加（https://github.com/settings/keys）

# 4. 修改本地配置
git remote set-url origin git@github.com:Limozknight/limxtron1lab-training.git

# 5. 推送
git push -u origin master
```

---

## ❓ 常见问题

**Q: 我的 GitHub 用户名是什么？**
- 你已经知道了：`Limozknight`
- 验证：https://github.com/Limozknight

**Q: 我应该用什么作为密码？**
- **不是你的 GitHub 密码！**
- 是你生成的 **Personal Access Token**

**Q: 生成的 token 丢了怎么办？**
- 需要重新生成新的 token（旧的无法恢复）

**Q: 多久需要重新生成 token？**
- 根据设置的过期时间
- 推荐设置 90 天

---

## ✅ 推送成功的标志

```bash
# 你应该看到类似这样的输出：

Enumerating objects: ...
Counting objects: ...
Compressing objects: ...
Writing objects: ...
Updating references: ...
To github.com:Limozknight/limxtron1lab-training.git
 * [new branch]      master -> master
Branch 'master' set to track remote branch 'master' from 'origin'.
```

然后打开你的仓库：
```
https://github.com/Limozknight/limxtron1lab-training
```

应该能看到所有代码已上传！

---

