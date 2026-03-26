param(
    [Parameter(Position = 0)]
    [string]$Message = "",

    [switch]$SkipPull,
    [switch]$NoPush,

    [ValidateRange(1, 2048)]
    [int]$MaxFileSizeMB = 100
)

$ErrorActionPreference = "Stop"

function Assert-GitRepo {
    $root = git rev-parse --show-toplevel 2>$null
    if (-not $root) {
        throw "当前目录不在 Git 仓库中。"
    }
    return $root.Trim()
}

function Get-CurrentBranch {
    $branch = git rev-parse --abbrev-ref HEAD 2>$null
    if (-not $branch) {
        throw "无法获取当前分支。"
    }
    return $branch.Trim()
}

function Assert-StagedFileSize([int]$limitMB) {
    $limitBytes = [int64]$limitMB * 1MB
    $stagedFiles = git diff --cached --name-only
    $oversized = @()

    foreach ($file in $stagedFiles) {
        if (-not [string]::IsNullOrWhiteSpace($file) -and (Test-Path $file)) {
            $size = (Get-Item $file).Length
            if ($size -ge $limitBytes) {
                $oversized += "{0} ({1:N2} MB)" -f $file, ($size / 1MB)
            }
        }
    }

    if ($oversized.Count -gt 0) {
        $details = $oversized -join ", "
        throw "检测到暂存区存在大文件（>= $limitMB MB），已终止提交：$details"
    }
}

$repoRoot = Assert-GitRepo
Set-Location $repoRoot

$branch = Get-CurrentBranch

if (-not $SkipPull) {
    Write-Host ">> 同步远端分支: origin/$branch"
    git fetch origin $branch
    $behindText = git rev-list --count "$branch..origin/$branch"
    $behind = 0
    if ($behindText -and ($behindText -match '^\d+$')) {
        $behind = [int]$behindText
    }

    if ($behind -gt 0) {
        Write-Host ">> 本地落后 $behind 个提交，执行 rebase 拉取"
        git pull --rebase origin $branch
    } else {
        Write-Host ">> 本地已是最新，无需 pull"
    }
}

Write-Host ">> 暂存所有改动"
git add -A

$pending = git status --porcelain
if (-not $pending) {
    Write-Host "没有可提交的改动。"
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = Read-Host "请输入本次提交说明"
}
if ([string]::IsNullOrWhiteSpace($Message)) {
    throw "提交说明不能为空。"
}

Assert-StagedFileSize -limitMB $MaxFileSizeMB

Write-Host ">> 提交: $Message"
git commit -m $Message

if (-not $NoPush) {
    Write-Host ">> 推送到 origin/$branch"
    git push origin $branch
    Write-Host ">> 上传完成。"
} else {
    Write-Host ">> 已提交，按参数要求跳过推送。"
}

git status --short --branch
