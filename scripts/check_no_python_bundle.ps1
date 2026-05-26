$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tauriConfigPath = Join-Path $repoRoot "src-tauri\tauri.conf.json"
$releaseRoot = Join-Path $repoRoot "src-tauri\target\release"
$bundleRoot = Join-Path $repoRoot "src-tauri\target\release\bundle"
$releaseExe = Join-Path $releaseRoot "pianke-desktop.exe"
$requiredOpenCvDlls = @(
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

function Test-ForbiddenPythonPath {
    param([Parameter(Mandatory = $true)][string]$PathText)

    $normalized = $PathText.Replace("\", "/").ToLowerInvariant()
    return (
        $normalized -match '(^|/|"|:|\s)python\.exe("|,|\s|$)' -or
        $normalized -match '(^|/|"|:|\s)app\.py("|,|\s|$)' -or
        $normalized -match '(^|/|"|:|\s)requirements-fast\.txt("|,|\s|$)' -or
        $normalized -match '(^|/)site-packages(/|$)' -or
        $normalized -match '(^|/|"|:|\s)site-packages(/|"|,|\s|$)' -or
        $normalized -match '(^|/|"|:|\s)pic_selecter(/|"|,|\s|$)' -or
        $normalized -match '(^|/|"|:|\s)binaries/python(/|"|,|\s|$)'
    )
}

function Add-ForbiddenHit {
    param(
        [System.Collections.Generic.List[string]]$Hits,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Value
    )

    if ($Hits.Count -ge 80) {
        return
    }

    if (Test-ForbiddenPythonPath -PathText $Value) {
        $Hits.Add("${Kind}: ${Value}")
    }
}

function Get-RepoRelativePath {
    param([Parameter(Mandatory = $true)][string]$FullName)

    if ($FullName.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $FullName.Substring($repoRoot.Length).TrimStart("\", "/")
    }
    return $FullName
}

function Scan-ArtifactTree {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Kind,
        [string[]]$ExcludeTopLevel = @()
    )

    if (-not (Test-Path -LiteralPath $Root)) {
        return $false
    }

    Get-ChildItem -LiteralPath $Root -Force | ForEach-Object {
        if ($ExcludeTopLevel -contains $_.Name) {
            return
        }

        Add-ForbiddenHit -Hits $hits -Kind $Kind -Value (Get-RepoRelativePath -FullName $_.FullName)
        if ($_.PSIsContainer) {
            Get-ChildItem -LiteralPath $_.FullName -Recurse -Force | ForEach-Object {
                Add-ForbiddenHit -Hits $hits -Kind $Kind -Value (Get-RepoRelativePath -FullName $_.FullName)
            }
        }
    }

    return $true
}

$hits = [System.Collections.Generic.List[string]]::new()

if (-not (Test-Path -LiteralPath $tauriConfigPath)) {
    throw "Tauri config file was not found: $tauriConfigPath"
}

Get-Content -LiteralPath $tauriConfigPath | ForEach-Object {
    Add-ForbiddenHit -Hits $hits -Kind "tauri config" -Value $_
}

if (-not (Scan-ArtifactTree -Root $bundleRoot -Kind "bundle output")) {
    Write-Host "Release bundle output was not found; skipping built artifact scan: $bundleRoot"
}

$releaseExcludes = @(".fingerprint", "build", "deps", "examples", "incremental")
if (-not (Scan-ArtifactTree -Root $releaseRoot -Kind "release output" -ExcludeTopLevel $releaseExcludes)) {
    Write-Host "Release output was not found; skipping release root scan: $releaseRoot"
}

if ($hits.Count -gt 0) {
    $message = "Python-related resources were found in release packaging output:`n" + ($hits -join "`n")
    if ($hits.Count -ge 80) {
        $message += "`n... output truncated after 80 hits ..."
    }
    Write-Error $message
    exit 1
}

if (Test-Path -LiteralPath $releaseExe) {
    $missingOpenCvDlls = @($requiredOpenCvDlls | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $releaseRoot $_))
    })
    if ($missingOpenCvDlls.Count -gt 0) {
        Write-Error ("OpenCV runtime DLLs are missing from release output:`n" + ($missingOpenCvDlls -join "`n"))
        exit 1
    }
}

Write-Host "Rust-only packaging check passed: no Python runtime, Flask worker, Python package, or missing OpenCV runtime resources were found."
