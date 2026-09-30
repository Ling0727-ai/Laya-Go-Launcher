# 启动 laya-trt。
#
#   .\run.ps1                       桌面端（默认）
#   .\run.ps1 -Server               只启动 HTTP API，不开窗口
#   .\run.ps1 -Engine <路径>        指定 engine
#   .\run.ps1 -Addr 0.0.0.0:8420    改监听地址
#   .\run.ps1 -NoCheck              跳过自检（确认环境没问题时更快）
#
# 它会先自检，再把 DLL 目录设进 PATH，最后启动。这两步顺序不能反：内核 DLL 是
# 加载期依赖，PATH 不对时进程会在 main() 之前就死掉，只留一个 0xC0000279。
#
# 手工启动等价于：
#   . .\env.ps1
#   .\cmd\layatrt-gui\build\bin\layatrt-gui.exe

param(
    [switch]$Server,
    [string]$Engine = "",
    [string]$Addr = "",
    [int]$Contexts = -1,
    [switch]$NoCheck,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

function Write-Bad($text) { Write-Host $text -ForegroundColor Red }

# ── 1. 自检 ─────────────────────────────────────────────────────────────────
if (-not $NoCheck) {
    $doctorExe = Join-Path $root 'doctor.exe'
    if (-not (Test-Path $doctorExe)) {
        if (Get-Command go -ErrorAction SilentlyContinue) {
            & go build -o $doctorExe ./cmd/layatrt-doctor 2>$null
        }
    }
    if (Test-Path $doctorExe) {
        & $doctorExe
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Bad "环境自检未通过，已中止启动。"
            Write-Host "按上面的 → 处理后重试，或加 -NoCheck 跳过。" -ForegroundColor DarkGray
            exit 1
        }
    } else {
        Write-Host "跳过自检（没找到 doctor.exe，且无法编译）" -ForegroundColor DarkGray
    }
}

# ── 2. 确保要启动的二进制存在 ───────────────────────────────────────────────
# 这一步必须在设 PATH 之前：env.ps1 会把 PATH 收窄到 DLL 目录，之后再找 go 就
# 找不到了。
#
# 编译这个项目必须开 CGO（internal/kernel 是 cgo 包）。关掉时 Go 只会报
# "build constraints exclude all Go files"，看不出是 cgo 的问题。
$env:CGO_ENABLED = '1'

# cgo 调用的是 gcc（或 go env CC 指定的那个）。装了 Visual Studio 不等于能用：
# cl.exe 默认不在 PATH 上。这里主动找一下常见的 mingw 安装位置，省掉用户自己配。
function Add-CompilerToPath {
    $cc = (& go env CC 2>$null)
    if ($cc -and (Get-Command $cc -ErrorAction SilentlyContinue)) { return $true }

    $candidates = @(
        "$env:ProgramFiles\llvm-mingw-*\bin",
        "$env:ProgramFiles\mingw64\bin",
        "C:\msys64\mingw64\bin",
        "C:\mingw64\bin",
        "$env:LOCALAPPDATA\Programs\mingw64\bin"
    )
    foreach ($pattern in $candidates) {
        $hits = Get-Item $pattern -ErrorAction SilentlyContinue
        foreach ($hit in $hits) {
            if (Test-Path (Join-Path $hit.FullName 'gcc.exe')) {
                $env:PATH = "$($hit.FullName);$env:PATH"
                Write-Host "找到 C 编译器: $($hit.FullName)\gcc.exe" -ForegroundColor DarkGray
                return $true
            }
        }
    }
    return $false
}

if ($Server) {
    $exe = Join-Path $root 'layatrt-server.exe'
    if (-not (Test-Path $exe)) {
        if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
            Write-Bad "没找到 $exe，也没有 go 可以编译它。"
            Write-Host "运行 .\setup.ps1 做完整准备。" -ForegroundColor DarkGray
            exit 1
        }
        $null = Add-CompilerToPath
        Write-Host "编译 layatrt-server…" -ForegroundColor DarkGray
        & go build -o $exe ./cmd/layatrt-server
        if ($LASTEXITCODE -ne 0) {
            Write-Bad "编译失败（错误见上）。"
            Write-Host "这个项目需要 CGO 和一个 C 编译器（cgo 默认找 gcc）。" -ForegroundColor DarkGray
            Write-Host "装 mingw-w64，或先跑 vcvars64.bat 再执行本脚本。" -ForegroundColor DarkGray
            exit 1
        }
    }
} else {
    $exe = Join-Path $root 'cmd\layatrt-gui\build\bin\layatrt-gui.exe'
    if (-not (Test-Path $exe)) {
        Write-Bad "没找到 $exe"
        Write-Host "构建它：" -ForegroundColor DarkGray
        Write-Host "  cd cmd\layatrt-gui; wails build" -ForegroundColor DarkGray
        Write-Host "或运行 .\setup.ps1 做完整准备。" -ForegroundColor DarkGray
        exit 1
    }
}

# ── 3. 设 PATH ──────────────────────────────────────────────────────────────
& (Join-Path $root 'env.ps1')

# ── 4. 启动 ─────────────────────────────────────────────────────────────────
if ($Server) {
    $args = @()
    if ($Engine) { $args += @('--engine', $Engine) }
    if ($Addr) { $args += @('--addr', $Addr) }
    if ($Contexts -ge 0) { $args += @('--contexts', $Contexts) }
    if ($ExtraArgs) { $args += $ExtraArgs }

    Write-Host ""
    Write-Host "启动 HTTP API…" -ForegroundColor Cyan
    & $exe @args
    exit $LASTEXITCODE
}

# 桌面端读配置文件；用参数覆盖时写进环境变量，因为 Wails 没有命令行透传。
if ($Engine) { $env:LAYA_TRT_ENGINE = $Engine }
if ($Addr) { $env:LAYA_TRT_HTTP_ADDR = $Addr }

Write-Host ""
Write-Host "启动桌面端…" -ForegroundColor Cyan
& $exe
exit $LASTEXITCODE
