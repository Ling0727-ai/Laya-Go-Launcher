# 一键准备构建环境。
#
#   .\setup.ps1              检查环境并编译内核 + Go 二进制
#   .\setup.ps1 -SkipKernel  只编译 Go 部分
#   .\setup.ps1 -Check       只做检查，不编译
#
# 这个脚本做的是「从 clone 到能跑」这一步。它会先自检，缺什么就停下来告诉你
# 装什么，而不是等到启动时抛一个退出码。

param(
    [switch]$SkipKernel,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

function Write-Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Write-Ok($text) { Write-Host "   $text" -ForegroundColor Green }
function Write-Bad($text) { Write-Host "   $text" -ForegroundColor Red }

# ── 1. 先自检 ───────────────────────────────────────────────────────────────
Write-Step "环境自检"

if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Bad "没找到 go，无法编译。装 Go 1.25+：https://go.dev/dl/"
    exit 1
}

# doctor 不链接内核，所以即使内核还没编译它也能跑起来做诊断。
$doctorExe = Join-Path $root 'doctor.exe'
if (-not (Test-Path $doctorExe)) {
    Write-Host "   编译自检工具…"
    & go build -o $doctorExe ./cmd/layatrt-doctor
    if ($LASTEXITCODE -ne 0) { Write-Bad "自检工具编译失败"; exit 1 }
}
& $doctorExe
$doctorOk = $LASTEXITCODE -eq 0

if ($Check) {
    exit $(if ($doctorOk) { 0 } else { 1 })
}

if (-not $doctorOk) {
    Write-Host ""
    Write-Host "自检未通过。按上面的 → 逐条处理后重试。" -ForegroundColor Yellow
    Write-Host "如果只是内核没编译，下一步会尝试编译它。" -ForegroundColor DarkGray
}

# ── 2. 编译内核 ─────────────────────────────────────────────────────────────
if (-not $SkipKernel) {
    Write-Step "编译内核 DLL"

    $haveVs = $false
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if ($vs) { $haveVs = $true }
    }

    if (-not $haveVs) {
        Write-Bad "没找到 Visual Studio 2022 (MSVC)。"
        Write-Host "   装 Visual Studio 2022 并勾选「使用 C++ 的桌面开发」：" -ForegroundColor DarkGray
        Write-Host "   https://visualstudio.microsoft.com/downloads/" -ForegroundColor DarkGray
        Write-Host "   只想跑程序、不需要自己编译内核的话，跳过这步并确保 build\bin\Release 下已有 DLL。" -ForegroundColor DarkGray
    } elseif (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Bad "没找到 cmake。装 CMake 3.18+：https://cmake.org/download/"
    } else {
        & (Join-Path $root 'build.ps1')
        if ($LASTEXITCODE -ne 0) { Write-Bad "内核编译失败"; exit 1 }
        Write-Ok "内核已编译"
    }
} else {
    Write-Step "跳过内核编译"
}

# ── 3. 编译 Go 二进制 ───────────────────────────────────────────────────────
Write-Step "编译 Go 二进制"
& go build ./...
if ($LASTEXITCODE -ne 0) { Write-Bad "Go 编译失败"; exit 1 }
Write-Ok "layatrt-gui / layatrt-server / layatrt-doctor 已就绪"

# ── 4. 再自检一次，确认状态 ─────────────────────────────────────────────────
Write-Step "最终自检"
& $doctorExe
$final = $LASTEXITCODE

Write-Host ""
if ($final -eq 0) {
    Write-Host "准备完成。启动：" -ForegroundColor Green
    Write-Host "  .\run.ps1            桌面端"
    Write-Host "  .\run.ps1 -Server    只启动 HTTP API"
    exit 0
}

Write-Host "构建已完成，但环境还有缺项（见上面的 → ）。" -ForegroundColor Yellow
Write-Host "最常见的是还没有 engine —— 需要先导出 ONNX 再编译，见 README 的「准备模型」。" -ForegroundColor DarkGray
exit 1
