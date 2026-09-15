<#
.SYNOPSIS
    Register the GREENCLOSE nightly forward-prediction runner as a Windows scheduled task.

.DESCRIPTION
    The forward log is the only evaluation in this project that provably cannot leak, because the
    prediction is written before the outcome exists. It is worth switching on early and never
    switching off.

    This script only REGISTERS a task that writes predictions to a parquet file. It cannot place an
    order: there is no broker code anywhere in this repository.

    Run it yourself when you want the schedule. Nothing in the build runs it for you.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_forward_task.ps1
    powershell -ExecutionPolicy Bypass -File scripts\register_forward_task.ps1 -Time 18:30 -WhatIf
    powershell -ExecutionPolicy Bypass -File scripts\register_forward_task.ps1 -Unregister
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Time = "18:15",                     # IST evening, comfortably after the 15:30 close
    [string]$TaskName = "GreenClose-Forward",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if ($Unregister) {
    if ($PSCmdlet.ShouldProcess($TaskName, "Unregister scheduled task")) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Unregistered $TaskName"
    }
    return
}

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { throw "uv is not on PATH. Install uv, or point the task at .venv\Scripts\python.exe instead." }

$logDir = Join-Path $root "data\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# cmd.exe wrapper so stdout and stderr land in a dated log the next morning can be read
$stamp = '%DATE:~-4%%DATE:~3,2%%DATE:~0,2%'
$logFile = Join-Path $logDir "forward_$stamp.log"
$inner = "`"$uv`" run python scripts\daily_forward.py --fetch"
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c $inner >> `"$logFile`" 2>&1" `
    -WorkingDirectory $root

$trigger  = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

$desc = "GREENCLOSE nightly forward prediction log. Predictions only: no broker, no orders, no money."

if ($PSCmdlet.ShouldProcess($TaskName, "Register daily task at $Time in $root")) {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Description $desc -Force | Out-Null
    Write-Host "Registered '$TaskName' to run daily at $Time"
    Write-Host "  working directory : $root"
    Write-Host "  command           : uv run python scripts\daily_forward.py --fetch"
    Write-Host "  logs              : $logDir"
    Write-Host ""
    Write-Host "Before the first run, pin a model:  uv run greenclose pin-model"
    Write-Host "Check it with:                      Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host "Run it once now with:               Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "Remove it with:                     .\scripts\register_forward_task.ps1 -Unregister"
}
