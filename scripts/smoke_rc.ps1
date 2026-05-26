$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host ""
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "RC smoke step failed: $Name"
    }
}

Invoke-Step "pianke-core tests" {
    cargo test --manifest-path crates\pianke-core\Cargo.toml
}

Invoke-Step "pianke-backend tests" {
    cargo test --manifest-path crates\pianke-backend\Cargo.toml
}

Invoke-Step "Tauri cargo check" {
    cargo check --manifest-path src-tauri\Cargo.toml
}

Invoke-Step "UI build check" {
    npm run build:ui
}

Invoke-Step "OpenCV release ORB smoke" {
    npm run test:opencv-release
}

if ($env:PIANKE_TYCOON_E2E -eq "1") {
    Invoke-Step "Tycoon mock E2E gated smoke" {
        cargo test --release --manifest-path crates\pianke-backend\Cargo.toml tycoon_mock_e2e_runs_with_complete_expert_component_when_configured -- --nocapture
    }
} else {
    Write-Host ""
    Write-Host "==> Tycoon mock E2E gated smoke"
    Write-Host "Skipping because PIANKE_TYCOON_E2E is not 1."
}

Invoke-Step "Rust-only release build" {
    npm run build
}

Invoke-Step "Release Rust backend smoke" {
    npm run smoke:release-rust
}

Invoke-Step "No-Python bundle scan" {
    npm run check:no-python-bundle
}

Invoke-Step "Whitespace diff check" {
    git diff --check
}

Write-Host ""
Write-Host "Rust-only RC smoke passed."
