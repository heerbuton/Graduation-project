param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Message
)

$ErrorActionPreference = "Stop"

git add -A

$pending = git status --porcelain
if (-not $pending) {
    Write-Host "没有可提交的改动。"
    exit 0
}

git commit -m $Message
git push
