# GitHub 上传项目教程

## 第一步：GitHub 新建仓库

去 [github.com](https://github.com) → 右上角 **+** → **New repository**

| 字段 | 填写内容 |
|------|----------|
| Repository name | 仓库名（建议英文，如 `fire-detection-yolo-vlm`） |
| Description | 可以先不填 |
| Visibility | **Public**（求职用，让面试官能看到） |
| Initialize options | **不要勾选** README / .gitignore / license（本地已有） |

点 **Create repository**，复制页面上的 HTTPS 地址：

```
https://github.com/sleepseagull/UAV-Fire-Collaboration.git
```

---

## 第二步：本地初始化并提交

在 VSCode 终端执行：

```bash
cd /d/毕设/1-fire-yolo-vlm
git init
git add .
git status
```

`git status` 会列出所有将被追踪的文件，确认没有大文件或不想要的内容。

> **关于 CRLF 警告**：Windows 上 Git 会把 LF 换行符自动转为 CRLF，这是默认行为，文件内容没有变化。上传到 GitHub 后仍保持 LF，其他人在 Linux/Mac 上拉取完全正常。

确认没问题后提交：

```bash
git commit -m "init: fire detection YOLO+VLM project"
```

> **首次提交前**：找到项目目录下的 `.git/config`，添加用户信息后保存：
> ```ini
> [user]
>     email = 17874632849@163.com
>     name = sleepseagull
> ```
> 首次 push 会弹出 GitHub 登录窗口，用浏览器授权即可。

```bash
git branch -M main
git remote add origin https://github.com/sleepseagull/UAV-Fire-Collaboration.git
git push -u origin main
```

---

## 第三步：处理大文件（>100MB）

如果模型文件超过 GitHub 单文件 100MB 限制会报错，需要用 Git LFS 处理。

### 为什么不能"先 commit 再追踪 LFS"？

GitHub 会扫描**所有历史提交**，不只是最新的。如果第一次 `git commit` 时大文件已经作为普通 blob 写入了历史，后来再追踪 LFS 也没用——旧 commit 里的大文件记录还在，push 会一直被拒。

**正确做法**：删掉旧的 `.git`，从头开始，确保第一个 commit 起大文件就走 LFS。

### 重新初始化

```powershell
cd "D:\毕设\1-fire-yolo-vlm"
Remove-Item -Recurse -Force .git 
（此语句在 **PowerShell** 中执行，不是 bash，目的是删除.git文件夹，想方便的话也可以手动删除）
```

追踪大文件类型（会自动生成 `.gitattributes`）：

```bash
git init
git lfs install
git lfs track "*.safetensors"
git lfs track "*.pt"
```

验证追踪是否生效：

```bash
git lfs ls-files
```

输出中文件名前有 `*` 号即为正确。

添加所有文件：

```bash
git add .
```

重新配置 `.git/config`（应包含 `[core]`、`[lfs]`、`[user]` 三节）：

```ini
[user]
    email = 17874632849@163.com
    name = sleepseagull
```

提交并推送：

```bash
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/sleepseagull/UAV-Fire-Collaboration.git
git push --set-upstream origin main
```

> **LFS 配额说明**：GitHub 免费账户提供 1GB LFS 存储 + 1GB/月带宽。
> `adapter_model.safetensors`（141MB）+ `optimizer.pt`（284MB）合计 425MB，存储够用。
> 每次 clone 会消耗带宽配额，注意不要超出 1GB/月。

---

## 日常维护

### 1、只更新某个文件

```bash
git add 文件路径
git commit -m "update: 描述改了什么"
git push
```

### 2、删除仓库中多余的文件

本地和仓库都删除：

```bash
git rm 文件路径
git commit -m "remove: 文件名"
git push
```

只从仓库删除，保留本地文件：

```bash
git rm --cached 文件路径
git rm -r --cached 文件夹路径 （加 -r 参数递归删除文件夹）
# 同时把该路径加入 .gitignore，防止下次又被追踪
git commit -m "remove: 文件名 from tracking"
git push
```

### 3、两边历史分叉了，直接 push 被拒
问题描述：

本地 main：有 a 个新提交（remove1, update3, update2...）

远端 origin/main：有 b 个本地没有的提交（Add files via upload，这是在 GitHub 网页上直接上传文件产生的）

运行这两条：
```bash
git pull --rebase origin main
git push
```
pull --rebase 会先把远端那个提交拉下来，再把你本地的 5 个提交接在后面，然后 push 就能成功了。