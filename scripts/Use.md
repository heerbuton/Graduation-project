# scripts 使用说明

本目录提供了一个自动提交并推送到 GitHub 的脚本：

- `git-sync.ps1`

## 1. 脚本功能

`git-sync.ps1` 会自动执行以下步骤：

1. 检查当前目录是否在 Git 仓库内
2. 默认先同步远端（`fetch`，必要时 `pull --rebase`）
3. `git add -A`
4. 检查是否有改动（没有改动就退出）
5. 检查暂存区是否有超大文件（默认拦截 `>=100MB`）
6. `git commit -m "<你的提交说明>"`
7. `git push origin <当前分支>`

## 2. 日常使用（推荐）

在项目根目录执行：

```powershell
git sync
```

说明：

- `git sync` 是已配置好的 Git 别名
- 会调用 `scripts/git-sync.ps1` 自动完成提交与推送
- 运行后会提示你输入“提交说明”，输入后回车即可继续

也支持直接传入说明（可选）：

```powershell
git sync "本次更新说明"
```

## 3. 常用参数

- `-SkipPull`：跳过同步远端步骤
- `-NoPush`：只提交不推送
- `-MaxFileSizeMB`：设置大文件拦截阈值（默认 `100`）

示例：

```powershell
# 只提交，不推送
git sync "临时提交" -NoPush

# 跳过 pull/rebase（例如离线场景）
git sync "本地改动" -SkipPull

# 将拦截阈值调成 120MB（不建议超过 GitHub 限制）
git sync "更新" -MaxFileSizeMB 120
```

## 4. 不用别名时的调用方式

如果你不想用别名，也可以直接运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/git-sync.ps1 "本次更新说明"
```

## 5. 首次克隆后（新机器）需要做的事

如果你在新电脑上克隆仓库，建议先执行一次别名配置：

```powershell
git config alias.sync "!pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/git-sync.ps1"
```

然后就可以继续使用：

```powershell
git sync "本次更新说明"
```

## 6. 常见提示

- 提示 `没有可提交的改动。`：说明当前没有文件变化，属于正常情况。
- 提示大文件拦截：请将模型权重等大文件移出提交，或加入 `.gitignore`。
- 推送失败提示认证问题：请确认本机已配置并可用 GitHub SSH Key。
