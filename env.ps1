# 把 Laya Go Launcher 运行所需的 DLL 目录加进 PATH。
#
#   . .\env.ps1          # 点源，作用于当前会话（推荐）
#
# 为什么需要这个文件：进程在加载期就解析这些 DLL，找不到会直接崩溃
# （退出码 0xC0000279），连 --help 都不会输出，也没有任何错误信息。
#
# 为什么顺序重要：系统上可能装了多个 CUDA（本机同时有 C:\CUDA 和
# CUDA Toolkit v13.3）。内核是针对某一个 CUDA 链接的，PATH 上先出现的
# 那个会被加载；顺序错了就会崩。所以这里按固定顺序显式排列，
# 并且只加必要的目录，不把整棵 CUDA 安装树塞进去。

$root = $PSScriptRoot

function Resolve-FirstDir {
    param([string[]]$Candidates)
    foreach ($c in $Candidates) {
        if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path }
    }
    return $null
}

# 顺序即优先级：内核 DLL → TensorRT → 内核链接的那个 CUDA。
# CUDA 的运行时 DLL 可能直接在 bin\ 下，也可能在 bin\x64\ 下。
$tensorrtRoot = if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { 'C:\TensorRT-10.16.0.72' }
$cudaRoot     = if ($env:CUDA_ROOT)     { $env:CUDA_ROOT }     else { 'C:\CUDA' }

$dirs = @()
$missing = @()

$kernelDir = Resolve-FirstDir @((Join-Path $root 'build\bin\Release'))
if ($kernelDir) { $dirs += $kernelDir } else { $missing += "kernel DLL: $(Join-Path $root 'build\bin\Release')" }

$trtDir = Resolve-FirstDir @((Join-Path $tensorrtRoot 'bin'), $tensorrtRoot)
if ($trtDir) { $dirs += $trtDir } else { $missing += "TensorRT: $tensorrtRoot\bin" }

$cudaDir = Resolve-FirstDir @(
    (Join-Path $cudaRoot 'bin\x64'),
    (Join-Path $cudaRoot 'bin'),
    $cudaRoot
)
if ($cudaDir) { $dirs += $cudaDir } else { $missing += "CUDA: $cudaRoot\bin\x64" }

if ($dirs.Count -eq 0) {
    Write-Warning "env.ps1: 没找到任何 DLL 目录。先运行 .\build.ps1，或设好 TENSORRT_ROOT / CUDA_ROOT。"
    return
}
if ($missing.Count -gt 0) {
    Write-Warning ("env.ps1: 缺少`n  " + ($missing -join "`n  "))
}

# 保留系统目录和工具链（go、node 等），避免把 PATH 收窄到只剩 DLL 目录。
# 但要把**其它** CUDA 副本剔除：内核是针对某一个 CUDA 链接的，PATH 上先出现
# 的那个会被加载，顺序错了就崩。这里只留我们指定的那一个。
$keep = @()
$seen = @{}
foreach ($d in ($env:PATH -split ';')) {
    if (-not $d) { continue }
    $key = $d.ToLowerInvariant().TrimEnd('\')
    if ($seen.ContainsKey($key)) { continue }

    # 丢掉别的 CUDA 运行时目录（保留我们选中的那个）。
    if ($key -ne $cudaDir.ToLowerInvariant().TrimEnd('\') -and
        $key -match 'cuda' -and
        (Get-ChildItem -Path $d -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue)) {
        continue
    }
    # 丢掉别的 TensorRT 目录。
    if ($key -ne $trtDir.ToLowerInvariant().TrimEnd('\') -and
        (Get-ChildItem -Path $d -Filter 'nvinfer_10.dll' -ErrorAction SilentlyContinue)) {
        continue
    }

    $seen[$key] = $true
    $keep += $d
}

$env:PATH = (($dirs + $keep) -join ';')

# ONNX CUDA / DirectML 需要各自的 onnxruntime 发布包（Assets/onnx/cuda12|cuda13|directml）。
# 本仓库没有自带时，复用 QualityScaler-go 的那一份；否则会落到 System32 里
# 不带 GPU provider 的 onnxruntime.dll。
if (-not $env:LAYA_TRT_ONNX_ROOT -and -not (Test-Path (Join-Path $root 'Assets\onnx'))) {
    $qs = Join-Path $env:USERPROFILE 'Downloads\QualityScaler-go'
    if (Test-Path (Join-Path $qs 'Assets\onnx')) {
        $env:LAYA_TRT_ONNX_ROOT = $qs
        Write-Host "env.ps1: LAYA_TRT_ONNX_ROOT=$qs" -ForegroundColor DarkGray
    }
}

# 逐项验证。
$checks = @(
    @{ name = 'qualityscaler_tensorrt.dll'; dir = $kernelDir },
    @{ name = 'nvinfer_10.dll';             dir = $trtDir },
    @{ name = 'cudart64_*.dll';             dir = $cudaDir }
)
$ok = $true
foreach ($check in $checks) {
    if (-not $check.dir) { $ok = $false; continue }
    if (-not (Get-ChildItem -Path $check.dir -Filter $check.name -ErrorAction SilentlyContinue)) {
        Write-Warning "env.ps1: $($check.dir) 下没找到 $($check.name)"
        $ok = $false
    }
}

if ($ok) {
    Write-Host 'env.ps1: PATH 就绪' -ForegroundColor Green
    foreach ($d in $dirs) { Write-Host "  $d" }
}
