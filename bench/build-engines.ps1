# Build laya TensorRT engines for several sequence-length presets.
#
#   pwsh -File bench\build-engines.ps1
#   pwsh -File bench\build-engines.ps1 -Presets 512,1024 -Precision fp32
#   pwsh -File bench\build-engines.ps1 -DryRun
#
# Why several presets instead of one
# ----------------------------------
# TensorRT sizes a plan's worst-case activation memory from the profile's
# --maxShapes. One plan built for 8192 tokens therefore reserves for 8192 even
# when every request is 512 tokens long, and that reservation is what decides how
# many execution contexts fit on the GPU. Measured on this machine:
#
#   seq max 1024, batch 1  ->    88 MiB per context
#   seq max 8192, batch 1  ->  4563 MiB per context
#
# So the preset is a real memory/latency trade, not a cosmetic choice: pick the
# smallest ceiling that covers the workload.
#
# Why the batch ceiling differs per preset
# ----------------------------------------
# Activation memory scales with the profile's worst case, which includes the
# batch dimension. Batch 8 at seq 8192 would ask for roughly 8 x 4563 MiB, far
# past this GPU, so the ceiling is reduced as the sequence ceiling grows. A
# plan whose batch profile is 1..1 keeps the engine single-row, which the Go side
# detects and handles correctly (see inference.chooseMaxBatch).
#
# The engines are written to -OutDir together with a manifest describing each
# one, so a caller can choose a preset without re-reading the plans.

param(
    # Comma-separated sequence ceilings. A string rather than [int[]] because
    # `pwsh -File` binds extra arguments positionally, so `-Presets 1024 2048`
    # would silently feed 2048 to the next parameter instead of the array.
    [string]$Presets = '512,1024,2048,8192',

    [string]$Onnx = 'C:\Users\lingxin\Documents\laya-ort-probe\laya_dyn.onnx',
    [string]$OutDir = '',

    [ValidateSet('fp16', 'fp32')]
    [string]$Precision = 'fp16',

    # Marker ceiling shared by every preset. The head scores one marker per
    # option, and the checkpoint's head budget is far below this, so 64 is
    # generous for all of them.
    [int]$MarkersMax = 64,

    # TensorRT builder optimisation level: higher builds slower and usually runs
    # faster. 1 keeps a full sweep practical; raise it for a production build.
    [int]$BuilderOptLevel = 1,

    [int]$WorkspaceMiB = 6000,

    # Build one two-profile plan instead of one plan per preset:
    #   profile 0: batch <= 8, seq <= -ShortSeq   (the trained context, batched)
    #   profile 1: batch  = 1, seq <= largest preset  (long inputs)
    # The kernel switches profile per run. One batched profile up to 8192 does
    # not build on a 12 GB GPU (its worst-case tactic asks for ~8 GB), and the
    # batch-1 8192 plan pays one forward pass per question and runs its short
    # requests on kernels tuned for the long ceiling. Measured p50 on the
    # autobench case set: 4.3 ms vs 11.2 ms for laya_s8192_fp16_b1.
    [switch]$MultiProfile,
    [int]$ShortSeq = 512,

    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root 'engines' }

$trtexec = if ($env:TENSORRT_ROOT) { Join-Path $env:TENSORRT_ROOT 'bin\trtexec.exe' }
           else { 'C:\TensorRT-10.16.0.72\bin\trtexec.exe' }

if (-not (Test-Path $trtexec)) { throw "trtexec not found at $trtexec (set TENSORRT_ROOT)" }
if (-not (Test-Path $Onnx)) { throw "ONNX model not found at $Onnx" }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

# Batch ceiling per sequence ceiling. The pair is what bounds worst-case
# activation memory, so it is stated as one table rather than two loose knobs.
$batchCeiling = @{
    512  = 8
    1024 = 8
    2048 = 4
    8192 = 1
}

# Sequence floor. A dynamic plan still has a minimum, and a request shorter than
# it is padded up by the Go side, so this only needs to be small.
$seqFloor = 64

function Get-BatchMax([int]$seq) {
    if ($batchCeiling.ContainsKey($seq)) { return $batchCeiling[$seq] }
    # Unknown preset: keep the pair conservative rather than guessing high.
    if ($seq -le 1024) { return 8 }
    if ($seq -le 2048) { return 4 }
    return 1
}

# optShapes steer tactic selection toward the shapes actually used: a handful of
# questions over a short-to-medium document, not the ceiling.
function Get-OptSeq([int]$seq) {
    $opt = [int]($seq / 2)
    if ($opt -lt 64) { $opt = 64 }
    if ($opt -gt 512) { $opt = 512 }
    return $opt
}

# The opt batch stays at 1 even when the profile allows more.
#
# Measured on the reference machine, an opt batch of 2 was slower for a
# single-question request (8.3 ms vs 7.1 ms p50) and no faster for eight
# questions (11.2 ms either way). Since a single-question call is the common
# interactive case and batching still engages through --maxShapes, the opt point
# is kept at the cheapest real request.
function Get-OptBatch([int]$batchMax) {
    return 1
}

function Get-OptMarkers([int]$markersMax) {
    if ($markersMax -ge 8) { return 8 }
    return $markersMax
}

function Format-Profile([int]$batch, [int]$seq, [int]$markers) {
    return "input_ids:${batch}x${seq},attention_mask:${batch}x${seq}," +
           "marker_pos:${batch}x${markers},marker_mask:${batch}x${markers}," +
           "qtype:${batch}"
}

# Parse the comma-separated preset list, rejecting anything that is not a
# positive integer rather than silently building the wrong profile.
$presetList = @()
foreach ($part in ($Presets -split ',')) {
    $t = $part.Trim()
    if (-not $t) { continue }
    $n = 0
    if (-not [int]::TryParse($t, [ref]$n) -or $n -lt 8) {
        throw "invalid preset '$t': expected a sequence ceiling of at least 8"
    }
    $presetList += $n
}
if ($presetList.Count -eq 0) { throw "no presets given" }
$presetList = $presetList | Sort-Object -Unique

if ($MultiProfile) {
    $longSeq = ($presetList | Measure-Object -Maximum).Maximum
    $shortBatch = Get-BatchMax $ShortSeq
    $optMarkers = Get-OptMarkers $MarkersMax
    $name = "laya_s${longSeq}_${Precision}_p2.engine"
    $outPath = Join-Path $OutDir $name
    $logPath = Join-Path $OutDir "$name.build.log"
    $args = @(
        "--onnx=$Onnx",
        "--saveEngine=$outPath",
        "--profile=0",
        "--minShapes=$(Format-Profile 1 $seqFloor 2)",
        "--optShapes=$(Format-Profile 1 ([int]($ShortSeq / 2)) $optMarkers)",
        "--maxShapes=$(Format-Profile $shortBatch $ShortSeq $MarkersMax)",
        "--profile=1",
        "--minShapes=$(Format-Profile 1 $seqFloor 2)",
        "--optShapes=$(Format-Profile 1 ([math]::Min(1024, $longSeq)) $optMarkers)",
        "--maxShapes=$(Format-Profile 1 $longSeq $MarkersMax)",
        "--builderOptimizationLevel=$BuilderOptLevel",
        "--memPoolSize=workspace:$WorkspaceMiB",
        "--skipInference"
    )
    if ($Precision -eq 'fp16') { $args += '--fp16' }
    Write-Host "[build] $name  (profile 0: b<=$shortBatch s<=$ShortSeq; profile 1: b=1 s<=$longSeq)" -ForegroundColor Cyan
    if ($DryRun) { Write-Host "        $trtexec $($args -join ' ')" -ForegroundColor DarkGray; exit 0 }
    if ((Test-Path $outPath) -and -not $Force) { Write-Host "[skip] $name already exists (use -Force)"; exit 0 }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $trtexec @args *> $logPath
    $code = $LASTEXITCODE
    # trtexec leaves a 0-byte file behind when the build fails.
    if ($code -ne 0 -or -not (Test-Path $outPath) -or (Get-Item $outPath).Length -eq 0) {
        if (Test-Path $outPath) { Remove-Item $outPath }
        Write-Host "        FAILED (exit $code); log: $logPath" -ForegroundColor Red
        exit 1
    }
    Write-Host "        ok: $([math]::Round((Get-Item $outPath).Length / 1MB, 1)) MB in $([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Green
    exit 0
}

$manifest = @()
$built = 0
$skipped = 0
$failed = @()

foreach ($seq in $presetList) {
    $batchMax = Get-BatchMax $seq
    $optSeq = Get-OptSeq $seq
    $optBatch = Get-OptBatch $batchMax
    $optMarkers = Get-OptMarkers $MarkersMax

    $name = "laya_s${seq}_${Precision}_b${batchMax}.engine"
    $outPath = Join-Path $OutDir $name
    $logPath = Join-Path $OutDir "$name.build.log"

    if ((Test-Path $outPath) -and -not $Force) {
        Write-Host "[skip] $name already exists (use -Force to rebuild)" -ForegroundColor DarkGray
        $skipped++
    }
    else {
        $args = @(
            "--onnx=$Onnx",
            "--saveEngine=$outPath",
            "--minShapes=$(Format-Profile 1 $seqFloor 2)",
            "--optShapes=$(Format-Profile $optBatch $optSeq $optMarkers)",
            "--maxShapes=$(Format-Profile $batchMax $seq $MarkersMax)",
            "--builderOptimizationLevel=$BuilderOptLevel",
            "--memPoolSize=workspace:$WorkspaceMiB"
        )
        if ($Precision -eq 'fp16') { $args += '--fp16' }

        Write-Host ""
        Write-Host "[build] $name" -ForegroundColor Cyan
        Write-Host "        seq <= $seq, batch <= $batchMax, markers <= $MarkersMax"
        Write-Host "        min $(Format-Profile 1 $seqFloor 2)"
        Write-Host "        opt $(Format-Profile $optBatch $optSeq $optMarkers)"
        Write-Host "        max $(Format-Profile $batchMax $seq $MarkersMax)"

        if ($DryRun) {
            Write-Host "        $trtexec $($args -join ' ')" -ForegroundColor DarkGray
            $skipped++
        }
        else {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            & $trtexec @args *> $logPath
            $code = $LASTEXITCODE
            $sw.Stop()

            if ($code -ne 0 -or -not (Test-Path $outPath)) {
                Write-Host "        FAILED (exit $code) after $([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Red
                Write-Host "        log: $logPath" -ForegroundColor Red
                $failed += $name
                continue
            }

            $sizeMB = [math]::Round((Get-Item $outPath).Length / 1MB, 1)
            Write-Host "        ok: $sizeMB MB in $([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Green
            $built++
        }
    }

    if (Test-Path $outPath) {
        $manifest += [ordered]@{
            name            = $name
            path            = $outPath
            precision       = $Precision
            seq_max         = $seq
            seq_min         = $seqFloor
            batch_max       = $batchMax
            markers_max     = $MarkersMax
            builder_opt     = $BuilderOptLevel
            size_mb         = [math]::Round((Get-Item $outPath).Length / 1MB, 1)
            built_at        = (Get-Item $outPath).LastWriteTimeUtc.ToString('o')
        }
    }
}

if (-not $DryRun -and $manifest.Count -gt 0) {
    $manifestPath = Join-Path $OutDir 'manifest.json'
    $doc = [ordered]@{
        generated_at = (Get-Date).ToUniversalTime().ToString('o')
        onnx         = $Onnx
        precision    = $Precision
        engines      = $manifest
    }
    $doc | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding utf8
    Write-Host ""
    Write-Host "wrote $manifestPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "built $built, skipped $skipped, failed $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host "failed: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
