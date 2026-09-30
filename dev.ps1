# 开发模式启动 GUI：先设好 PATH，再跑 wails dev。
#
#   .\dev.ps1                    # 热重载开发
#   .\dev.ps1 -SkipBindings      # 跳过绑定生成（前端结构没变时更快）
#   .\dev.ps1 -DevPort 34120     # 换 Wails dev server 端口
#   .\dev.ps1 -WebPort 5189      # 换前端 Vite 端口
#
# 为什么要包一层：`wails dev` 生成绑定的方式是**编译并运行**这个程序，而程序在
# main() 之前就要解析 qualityscaler_tensorrt.dll。这是加载期导入，不是运行期，
# 所以 PATH 不对时进程会静默崩溃（exit 0xc0000279），Wails 只报
# "exit status 0xc0000279"，看不出是 DLL 的问题。
#
# 端口：默认 5188（前端）和 34116（Wails dev server），都刻意避开 Vite 默认的
# 5173 和 Wails 默认的 34115 —— 同一台机器上常有别的项目占着那两个。
#
# 手工跑 `wails dev` 时，必须先 `. .\env.ps1` 设好 PATH。

param(
    [switch]$SkipBindings,
    [int]$DevPort = 34116,
    [int]$WebPort = 5188,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

. (Join-Path $root 'env.ps1')

if (-not (Get-Command wails -ErrorAction SilentlyContinue)) {
    Write-Error "wails 未安装。先运行：go install github.com/wailsapp/wails/v2/cmd/wails@latest"
    exit 1
}

$gui = Join-Path $root 'cmd\layatrt-gui'
if (-not (Test-Path $gui)) {
    Write-Error "找不到 $gui"
    exit 1
}

# Fail early and clearly if a port is taken, instead of letting Wails/Vite pick
# something else and make the printed URL wrong.
foreach ($check in @(@{Port = $DevPort; What = 'Wails dev server' },
                     @{Port = $WebPort; What = '前端 Vite' })) {
    $inUse = Get-NetTCPConnection -LocalPort $check.Port -State Listen -ErrorAction SilentlyContinue
    if ($inUse) {
        $owner = Get-Process -Id $inUse[0].OwningProcess -ErrorAction SilentlyContinue
        Write-Host "$($check.What) 端口 $($check.Port) 已被 $($owner.ProcessName) (PID $($inUse[0].OwningProcess)) 占用。" -ForegroundColor Yellow
        Write-Host "换一个：.\dev.ps1 -DevPort 34117 -WebPort 5189" -ForegroundColor DarkGray
        exit 1
    }
}

# The Vite side reads its port from this variable.
$env:LAYATRT_DEV_PORT = "$WebPort"

$wailsArgs = @('dev', '-devserver', "localhost:$DevPort")
if ($SkipBindings) { $wailsArgs += '-skipbindings' }
if ($ExtraArgs) { $wailsArgs += $ExtraArgs }

Write-Host "dev.ps1: 前端 http://localhost:$WebPort   dev server http://localhost:$DevPort" -ForegroundColor Cyan
Write-Host "dev.ps1: wails $($wailsArgs -join ' ')" -ForegroundColor DarkGray

Push-Location $gui
try {
    & wails @wailsArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
