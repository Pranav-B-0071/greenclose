<#
.SYNOPSIS
    Run every remaining GREENCLOSE phase end to end and regenerate the report.

.DESCRIPTION
    One command, roughly 2 to 2.5 hours on a 16-core laptop. Safe to interrupt: the model ladder
    caches its predictions, so re-running skips whatever already finished. Baselines and sweeps
    restart from scratch if interrupted.

    Order is deliberate. The ladder comes first because it is the main experiment, and the 1,000
    random simulations come last because they are the longest single step.

    Paper only. Nothing in here can place an order.

.PARAMETER Jobs
    Worker processes. Default: half the logical cores, which leaves the machine usable.

.PARAMETER B0Sims
    Random simulations for the B0 null distribution. The brief asks for 1000.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 -Jobs 12 -B0Sims 1000
    powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 -SkipBaselines
#>
[CmdletBinding()]
param(
    [int]$Jobs = [Math]::Max(2, [Environment]::ProcessorCount / 2),
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

function Step([string]$Name, [scriptblock]$Body) {
    $t0 = Get-Date
    Write-Host ""
    Write-Host "=== $Name  (started $($t0.ToString('HH:mm:ss')))" -ForegroundColor Cyan
    "=== $Name  started $($t0.ToString('o'))" | Add-Content $log
    & $Body 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        throw "$Name failed with exit code $LASTEXITCODE. See $log"
    }
    $mins = [Math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Write-Host "=== $Name done in $mins min" -ForegroundColor Green
    "=== $Name done in $mins min" | Add-Content $log
}

$started = Get-Date
Write-Host "GREENCLOSE full run. jobs=$Jobs  B0 sims=$B0Sims" -ForegroundColor Yellow
Write-Host "log: $log"

if (-not $SkipLadder) {
    Step "P7 model ladder, rungs 3-8, 5 seeds, plus the H2 experiment (~50 min)" {
        uv run greenclose ladder --rungs 3-8 --jobs $Jobs
    }
}

if (-not $SkipSweeps) {
    Step "P8 decision-layer sweeps: hold, K, cost multiplier, product, rebalance (~10 min)" {
        uv run greenclose sweep --group decision
    }
    Step "P8 retrain sweeps: C2C/C2O/O2C horizons, expanding vs 2-year rolling (~15 min)" {
        uv run greenclose sweep --group retrain
    }
}

if (-not $SkipBaselines) {
    Step "P5 baselines with the full B0 null distribution, $B0Sims simulations (~70 min)" {
        uv run greenclose baselines --b0-sims $B0Sims --jobs $Jobs
    }
}

Step "P9 pin the forward model and write tonight's prediction log" {
    uv run greenclose pin-model --head lgbm_huber --reason "pinned by run_full.ps1"
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
