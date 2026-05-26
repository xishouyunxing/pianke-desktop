$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$defaultVcpkgRoot = Join-Path $env:USERPROFILE "vcpkg"
$vcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { $defaultVcpkgRoot }
$triplet = if ($env:VCPKGRS_TRIPLET) { $env:VCPKGRS_TRIPLET } else { "x64-windows" }
$llvmBin = if ($env:LLVM_BIN) { $env:LLVM_BIN } else { "C:\Program Files\LLVM\bin" }

if (-not (Test-Path -LiteralPath $vcvars)) {
    throw "Visual Studio vcvars64.bat was not found: $vcvars"
}
if (-not (Test-Path -LiteralPath (Join-Path $vcpkgRoot "vcpkg.exe"))) {
    throw "vcpkg.exe was not found: $vcpkgRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $llvmBin "clang.exe"))) {
    throw "clang.exe was not found: $llvmBin"
}

$escapedRepo = $repoRoot.Replace('"', '\"')
$escapedVcvars = $vcvars.Replace('"', '\"')
$escapedVcpkg = $vcpkgRoot.Replace('"', '\"')
$escapedTriplet = $triplet.Replace('"', '\"')
$escapedLlvm = $llvmBin.Replace('"', '\"')

$command = @"
call "$escapedVcvars"
if errorlevel 1 exit /b %errorlevel%
set "VCPKG_ROOT=$escapedVcpkg"
set "VCPKGRS_DYNAMIC=1"
set "VCPKGRS_TRIPLET=$escapedTriplet"
set "LIBCLANG_PATH=$escapedLlvm"
set "PATH=$escapedLlvm;%PATH%"
cd /d "$escapedRepo"
cargo test --release --manifest-path crates\pianke-backend\Cargo.toml --features opencv-orb opencv_orb_inliers_detect_repeated_scene_geometry -- --nocapture
"@

$cmdFile = Join-Path $env:TEMP ("pianke-opencv-test-" + [guid]::NewGuid().ToString("N") + ".cmd")
try {
    Set-Content -LiteralPath $cmdFile -Value $command -Encoding ASCII
    & cmd.exe /d /s /c "`"$cmdFile`""
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Remove-Item -LiteralPath $cmdFile -Force -ErrorAction SilentlyContinue
}
