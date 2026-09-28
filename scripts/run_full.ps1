<#
.SYNOPSIS
    Run every remaining GREENCLOSE phase end to end and regenerate the report.

.DESCRIPTION
    One command. Measured on 2026-09-16 on a 16-core, 15 GB laptop: 14 min ladder with the model
    fits already cached, 5 min decision sweeps, 3.5 h retrain sweeps, ~30 min baselines. Budget
    four to five hours, or pass -SkipSweeps once the sweeps are done, which is most of it. Memory
    rather than cores is usually what limits the pool. Safe to interrupt: the model ladder caches its
    predictions, so re-running skips whatever already finished. Baselines and sweeps restart from
    scratch if interrupted.

    The retrain sweeps are slow because they are the only step that refits models -- six cells, each a
    full monthly-retrain walk-forward, and nothing in them can be cached from the ladder.

    Order is deliberate. The ladder comes first because it is the main experiment, and the 1,000
    random simulations come last because they are the longest single step.

    Paper only. Nothing in here can place an order.

.PARAMETER Jobs
    Worker processes. Default 0, meaning choose automatically: each worker needs about 0.6 GB, so
    the pool is sized from free memory first and cores second, keeping 1 GB of headroom. Per-worker
    library threads are capped automatically, so more workers does not mean more thread thrashing.
    Pass a number to override.

.PARAMETER B0Sims
    Random simulations for the B0 null distribution. The brief asks for 1000.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 -Jobs 12 -B0Sims 1000
    powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 -SkipBaselines
#>
[CmdletBinding()]
param(
    [int]$Jobs = 0,            # 0 = choose from free memory and cores, see below
    [int]$B0Sims = 1000,
    [switch]$SkipLadder,
    [switch]$SkipSweeps,
    [switch]$SkipBaselines
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "data\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $logDir "run_full_$stamp.log"

# Tee-Object has no -Encoding in Windows PowerShell and writes UTF-16, while Add-Content wrote the
# banners as ANSI: the first run's log came out half readable and half interleaved with NULs. One
# BOM-less UTF-8 writer for every line keeps the log one encoding, and appending directly is faster
# than a cmdlet per line.
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
function Write-Log([string]$Line) {
    [System.IO.File]::AppendAllText($log, $Line + [Environment]::NewLine, $script:Utf8NoBom)
}

function Step([string]$Name, [scriptblock]$Body) {
    $t0 = Get-Date
    Write-Host ""
    Write-Host "=== $Name  (started $($t0.ToString('HH:mm:ss')))" -ForegroundColor Cyan
    Write-Log "=== $Name  started $($t0.ToString('o'))"

    # Windows PowerShell wraps every line a native program writes to stderr in an ErrorRecord, so
    # under ErrorActionPreference=Stop one library warning on stderr would abort an hour-long run
    # that was otherwise going fine. The process exit code is the only honest success signal, so
    # relax the preference around the call and check that instead. "$_" flattens each record back to
    # the line the program actually printed, which is what belongs in the log.
    $global:LASTEXITCODE = 0
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Body 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            Write-Log $line
        }
    } finally {
        $ErrorActionPreference = $prev
    }

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE. See $log"
    }
    $mins = [Math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Write-Host "=== $Name done in $mins min" -ForegroundColor Green
    Write-Log "=== $Name done in $mins min"
}

# Each worker holds its own feature panel and day slices: about 0.5 GB resident. Memory, not cores,
# is what limits this machine, and a worker that swaps is far slower than one that waits. So size the
# pool from free memory first and cores second.
$WorkerGB = 0.6
if ($Jobs -le 0) {
    $freeGB  = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $byMem   = [Math]::Floor(($freeGB - 1.0) / $WorkerGB)      # keep 1 GB headroom for the OS
    $byCores = [Environment]::ProcessorCount - 2
    $Jobs    = [int][Math]::Max(2, [Math]::Min($byMem, $byCores))
    Write-Host ("free memory {0:N1} GB -> {1} workers by memory, {2} by cores; using {3}" -f $freeGB, $byMem, $byCores, $Jobs)
    if ($byMem -lt $byCores) {
        Write-Host "  (closing browsers and editors before a long run buys you more workers)" -ForegroundColor DarkGray
    }
}

$started = Get-Date
Write-Host "GREENCLOSE full run. jobs=$Jobs  B0 sims=$B0Sims" -ForegroundColor Yellow
Write-Host "log: $log"

if (-not $SkipLadder) {
    Step "P7 model ladder, rungs 3-8, 5 seeds, plus the H2 experiment (14 min cached, 1.7 h cold)" {
        uv run greenclose ladder --rungs 3-8 --jobs $Jobs
    }
}

if (-not $SkipSweeps) {
    Step "P8 decision-layer sweeps: hold, K, cost multiplier, product, rebalance (~5 min)" {
        uv run greenclose sweep --group decision
    }
    Step "P8 retrain sweeps: C2C/C2O/O2C horizons, expanding vs 2-year rolling (~3.5 h: it refits)" {
        uv run greenclose sweep --group retrain
    }
}

if (-not $SkipBaselines) {
    Step "P5 baselines with the full B0 null distribution, $B0Sims simulations (~30 min)" {
        uv run greenclose baselines --b0-sims $B0Sims --jobs $Jobs
    }
}

Step "P9 pin the forward model" {
    uv run greenclose pin-model --head lgbm_huber --reason "pinned by run_full.ps1"
}

Step "P9 write tonight's forward prediction log" {
    uv run greenclose forward
}

Step "P9 report" {
    uv run greenclose report
}

$mins = [Math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Host ""
Write-Host "FULL RUN COMPLETE in $mins min" -ForegroundColor Yellow
Write-Host "  report   : $(Join-Path $root 'reports')  (open the newest .html)"
Write-Host "  evidence : research\research_log.jsonl, research\ladder.json, research\leakage_suite.json"
Write-Host ""
Write-Host "The locked holdout is NOT part of this run. It may be evaluated exactly once, when you"
Write-Host "are ready to stop changing things:"
Write-Host "  uv run greenclose holdout --confirm" -ForegroundColor Yellow
