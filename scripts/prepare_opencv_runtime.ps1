$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defaultVcpkgRoot = Join-Path $env:USERPROFILE "vcpkg"
$vcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { $defaultVcpkgRoot }
$triplet = if ($env:VCPKGRS_TRIPLET) { $env:VCPKGRS_TRIPLET } else { "x64-windows" }
$sourceDir = Join-Path $vcpkgRoot "installed\$triplet\bin"
$targetDir = Join-Path $repoRoot "src-tauri\opencv-runtime"

$requiredDlls = @(
    "opencv_core4.dll",
    "opencv_imgproc4.dll",
    "opencv_features2d4.dll",
    "opencv_calib3d4.dll",
    "opencv_flann4.dll",
    "opencv_imgcodecs4.dll",
    "opencv_dnn4.dll",
    "opencv_highgui4.dll",
    "opencv_ml4.dll",
    "opencv_objdetect4.dll",
    "opencv_photo4.dll",
    "opencv_stitching4.dll",
    "opencv_video4.dll",
    "opencv_videoio4.dll",
    "abseil_dll.dll",
    "jpeg62.dll",
    "liblzma.dll",
    "libpng16.dll",
    "libprotobuf.dll",
    "libprotobuf-lite.dll",
    "libsharpyuv.dll",
    "libwebp.dll",
    "libwebpdecoder.dll",
    "libwebpdemux.dll",
    "libwebpmux.dll",
    "tiff.dll",
    "turbojpeg.dll",
    "z.dll"
)

if (-not (Test-Path -LiteralPath $sourceDir)) {
    throw "OpenCV runtime source directory was not found: $sourceDir"
}

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$copied = 0
$totalBytes = 0L
foreach ($dll in $requiredDlls) {
    $source = Join-Path $sourceDir $dll
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required OpenCV runtime DLL was not found: $source"
    }

    $target = Join-Path $targetDir $dll
    Copy-Item -LiteralPath $source -Destination $target -Force
    $item = Get-Item -LiteralPath $target
    $copied += 1
    $totalBytes += $item.Length
}

$totalMb = [math]::Round($totalBytes / 1MB, 2)
Write-Host "Prepared $copied OpenCV runtime DLLs in $targetDir ($totalMb MB)."
