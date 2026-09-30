# Build the native kernels.
#
#   .\build.ps1
#   .\build.ps1 -TensorRTRoot C:\TensorRT-10.16.0.72 -CudaRoot C:\CUDA
#
# Produces two libraries into build\bin\<Config>:
#
#   qualityscaler_tensorrt.dll   the TensorRT kernel (needs TensorRT + CUDA)
#   layatrt_onnx.dll             the ONNX Runtime bridge (needs only the ORT header)
#
# Requires: Visual Studio 2022 (MSVC), CMake, CUDA Toolkit, TensorRT 10.x.
# The ONNX bridge is skipped, with a warning, when the ORT C API header is
# missing; the TensorRT kernel is still built and usable.

param(
    [string]$TensorRTRoot = "C:\TensorRT-10.16.0.72",
    [string]$CudaRoot     = "C:\CUDA",
    [string]$BuildDir     = "build",
    [string]$Config       = "Release",
    [string]$ORTHeaderDir = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not (Test-Path (Join-Path $TensorRTRoot "include\NvInfer.h"))) {
    throw "NvInfer.h not found under $TensorRTRoot\include"
}
if (-not (Test-Path (Join-Path $CudaRoot "include\cuda_runtime_api.h"))) {
    throw "cuda_runtime_api.h not found under $CudaRoot\include"
}

Write-Host "[build] TensorRT : $TensorRTRoot"
Write-Host "[build] CUDA     : $CudaRoot"

$cmakeArgs = @(
    "-S", $root,
    "-B", (Join-Path $root $BuildDir),
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DTENSORRT_ROOT=$TensorRTRoot",
    "-DCUDA_ROOT=$CudaRoot"
)
if ($ORTHeaderDir) {
    $cmakeArgs += "-DORT_HEADER_DIR=$ORTHeaderDir"
}

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build (Join-Path $root $BuildDir) --config $Config
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

# Verify what the build was supposed to produce. The TensorRT kernel keeps the
# upstream name so the existing cgo LDFLAGS (-lqualityscaler_tensorrt) resolve;
# an earlier version of this script looked for laya_trt.dll, which CMake never
# emitted, so a successful build reported a missing output.
$outDir = Join-Path $root "$BuildDir\bin\$Config"
$expected = @("qualityscaler_tensorrt.dll")
$optional = @("layatrt_onnx.dll")

foreach ($name in $expected) {
    $path = Join-Path $outDir $name
    if (-not (Test-Path $path)) { throw "expected output missing: $path" }
    Write-Host "[build] OK -> $path" -ForegroundColor Green
}

foreach ($name in $optional) {
    $path = Join-Path $outDir $name
    if (Test-Path $path) {
        Write-Host "[build] OK -> $path" -ForegroundColor Green
    } else {
        Write-Warning "[build] $name was not built; the ONNX kernel will be unavailable. " +
            "Place onnxruntime_c_api.h in third_party\onnxruntime or pass -ORTHeaderDir."
    }
}
