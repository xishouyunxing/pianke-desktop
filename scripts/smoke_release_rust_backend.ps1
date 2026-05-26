$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$releaseExe = Join-Path $repoRoot "src-tauri\target\release\pianke-desktop.exe"
$bundleCheck = Join-Path $PSScriptRoot "check_no_python_bundle.ps1"

if (-not (Test-Path -LiteralPath $releaseExe)) {
    throw "Release executable was not found. Run npm run build first: $releaseExe"
}

& powershell -ExecutionPolicy Bypass -File $bundleCheck

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $releaseExe
$startInfo.WorkingDirectory = Split-Path -Parent $releaseExe
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.EnvironmentVariables["PIANKE_BACKEND"] = "python"

$process = [System.Diagnostics.Process]::Start($startInfo)
if ($null -eq $process) {
    throw "Failed to start release executable: $releaseExe"
}

try {
    Start-Sleep -Seconds 6
    $process.Refresh()
    if ($process.HasExited) {
        throw "Release executable exited early with code $($process.ExitCode)"
    }

    $descendants = Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $process.Id
    }
    $pythonChildren = @($descendants | Where-Object {
        $_.Name -ieq "python.exe" -or $_.Name -ieq "pythonw.exe" -or $_.CommandLine -match "app\.py"
    })

    if ($pythonChildren.Count -gt 0) {
        $details = $pythonChildren | ForEach-Object {
            "pid=$($_.ProcessId) name=$($_.Name) command=$($_.CommandLine)"
        }
        throw ("Release smoke failed: Python backend child process was started.`n" + ($details -join "`n"))
    }

    Write-Host "Release Rust backend smoke passed: app stayed running and no Python backend child process was started."
} finally {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        $process.WaitForExit(5000) | Out-Null
    }
}
