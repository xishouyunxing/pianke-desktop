$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Read-JsonVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if (-not $json.version) {
        throw "$Label does not contain a version field."
    }
    return [string]$json.version
}

function Read-JsonVersionField {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $content = Get-Content -LiteralPath $Path -Raw
    $versionMatch = [regex]::Match(
        $content,
        '(?m)"version"\s*:\s*"(?<version>[^"]+)"'
    )
    if (-not $versionMatch.Success) {
        throw "$Label does not contain a version field."
    }

    return $versionMatch.Groups["version"].Value
}

function Read-CargoPackageVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $content = Get-Content -LiteralPath $Path -Raw
    $packageMatch = [regex]::Match(
        $content,
        '(?ms)^\[package\]\s*(?<body>.*?)(?:^\[|\z)'
    )
    if (-not $packageMatch.Success) {
        throw "$Label does not contain a [package] section."
    }

    $versionMatch = [regex]::Match(
        $packageMatch.Groups["body"].Value,
        '(?m)^\s*version\s*=\s*"(?<version>[^"]+)"'
    )
    if (-not $versionMatch.Success) {
        throw "$Label does not contain a package version."
    }

    return $versionMatch.Groups["version"].Value
}

$versions = [ordered]@{
    "package.json" = Read-JsonVersion `
        -Path (Join-Path $repoRoot "package.json") `
        -Label "package.json"
    "src-tauri/tauri.conf.json" = Read-JsonVersionField `
        -Path (Join-Path $repoRoot "src-tauri\tauri.conf.json") `
        -Label "src-tauri/tauri.conf.json"
    "src-tauri/Cargo.toml" = Read-CargoPackageVersion `
        -Path (Join-Path $repoRoot "src-tauri\Cargo.toml") `
        -Label "src-tauri/Cargo.toml"
    "crates/pianke-backend/Cargo.toml" = Read-CargoPackageVersion `
        -Path (Join-Path $repoRoot "crates\pianke-backend\Cargo.toml") `
        -Label "crates/pianke-backend/Cargo.toml"
}

$expected = $versions.Values | Select-Object -First 1
$mismatches = @()
foreach ($entry in $versions.GetEnumerator()) {
    if ($entry.Value -ne $expected) {
        $mismatches += "$($entry.Key)=$($entry.Value)"
    }
}

if ($mismatches.Count -gt 0) {
    Write-Host "Version mismatch detected:"
    foreach ($entry in $versions.GetEnumerator()) {
        Write-Host "  $($entry.Key): $($entry.Value)"
    }
    throw "All release-facing versions must match before packaging."
}

Write-Host "Version consistency OK: $expected"
