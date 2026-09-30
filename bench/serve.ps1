# Start layatrt-server on a private port for benchmarking.
#
#   pwsh -File bench\serve.ps1 -Engine engines\laya_s512_fp16_b8.engine
#
# The service is started as a background job on a port that is not the default,
# so it cannot disturb an instance someone else is running. It is stopped with
# bench\stop-serve.ps1.
#
# Why a script instead of an inline command: the DLL directories must be on PATH
# before the process starts (the kernel is a load-time dependency, and a missing
# DLL kills the process before main() with exit code 0xC0000279 and no message).
# env.ps1 does that; this wraps it with the flags a benchmark needs.

param(
    [string]$Engine = 'engines\laya_s512_fp16_b8.engine',
    [int]$Port = 8471,
    [int]$Contexts = 0,
    [int]$MaxLen = 0,
    [int]$HeadMaxLen = 0,
    [switch]$Diag
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $Engine)) { throw "engine not found: $Engine" }

$env:CGO_ENABLED = '1'
. (Join-Path $root 'env.ps1') | Out-Null

$exe = Join-Path $root 'layatrt-server.exe'

# Rebuild when the binary is older than any Go source it is built from.
#
# Checking only for the file's existence is a trap that was hit while writing
# this: a binary from the previous day silently served the old behaviour, and the
# benchmark measured code that was no longer in the tree. Rebuilding on a stale
# timestamp is cheap (Go caches unchanged packages) and removes the trap.
function Test-BinaryStale {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $true }
    $exeTime = (Get-Item $Path).LastWriteTimeUtc
    $newest = Get-ChildItem -Path (Join-Path $root 'internal'), (Join-Path $root 'cmd') `
        -Recurse -Filter '*.go' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if ($null -eq $newest) { return $false }
    return $newest.LastWriteTimeUtc -gt $exeTime
}

if (Test-BinaryStale $exe) {
    Write-Host "building layatrt-server (missing or older than the sources)..." -ForegroundColor DarkGray
    & go build -o $exe ./cmd/layatrt-server
    if ($LASTEXITCODE -ne 0) { throw "go build failed" }
}

$args = @('--engine', (Resolve-Path $Engine).Path, '--addr', "127.0.0.1:$Port")
if ($Contexts -gt 0) { $args += @('--contexts', $Contexts) }
if ($MaxLen -gt 0) { $args += @('--max-len', $MaxLen) }
if ($HeadMaxLen -gt 0) { $args += @('--head-max-len', $HeadMaxLen) }
if ($Diag) { $args += '--diagnostics' }

Write-Host "starting layatrt-server on 127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "  engine: $((Resolve-Path $Engine).Path)"
& $exe @args
