# scripts 使用说明

本目录提供了一个自动提交并推送到 GitHub 的脚本：

- `git-sync.ps1`

## 1. 脚本功能

`git-sync.ps1` 会自动执行以下步骤：

1. `git add -A`
2. 检查是否有改动（没有改动就退出）
3. `git commit -m "<你的提交说明>"`
4. `git push`

## 2. 日常使用（推荐）

在项目根目录执行：

```powershell
git sync "本次更新说明"
```

说明：

- `git sync` 是已配置好的 Git 别名
- 会调用 `scripts/git-sync.ps1` 自动完成提交与推送

## 3. 不用别名时的调用方式

如果你不想用别名，也可以直接运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/git-sync.ps1 "本次更新说明"
```

## 4. 首次克隆后（新机器）需要做的事

如果你在新电脑上克隆仓库，建议先执行一次别名配置：

```powershell
git config alias.sync "!pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/git-sync.ps1"
```

然后就可以继续使用：

```powershell
git sync "本次更新说明"
```

## 5. 常见提示

- 提示 `没有可提交的改动。`：说明当前没有文件变化，属于正常情况。
- 推送失败提示认证问题：请确认本机已配置并可用 GitHub SSH Key。
