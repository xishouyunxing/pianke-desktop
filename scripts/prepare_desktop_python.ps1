param(
    [string]$PythonVersion = "3.10.11",
    [string]$Mirror = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$CacheDir = Join-Path $Root ".desktop-python-cache"
$OutDir = Join-Path $Root "src-tauri\binaries\python"
$Requirements = Join-Path $Root "requirements-fast.txt"

if (-not (Test-Path $Requirements)) {
    throw "Missing requirements-fast.txt at $Requirements"
}

if ((Test-Path $OutDir) -and -not $Force) {
    Write-Host "Bundled Python already exists: $OutDir"
    Write-Host "Use -Force to rebuild it."
    exit 0
}

if (Test-Path $OutDir) {
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force $CacheDir | Out-Null
New-Item -ItemType Directory -Force $OutDir | Out-Null

$ZipName = "python-$PythonVersion-embed-amd64.zip"
$ZipPath = Join-Path $CacheDir $ZipName
$BaseUrl = if ($Mirror) { $Mirror.TrimEnd("/") } else { "https://www.python.org/ftp/python/$PythonVersion" }
$ZipUrl = "$BaseUrl/$ZipName"

if (-not (Test-Path $ZipPath)) {
    Write-Host "Downloading $ZipUrl"
    Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath
}

Write-Host "Extracting Python $PythonVersion to $OutDir"
Expand-Archive -Path $ZipPath -DestinationPath $OutDir -Force

$Pth = Get-ChildItem -LiteralPath $OutDir -Filter "python*._pth" | Select-Object -First 1
if (-not $Pth) {
    throw "Unable to find python*._pth in $OutDir"
}
$PthText = Get-Content -LiteralPath $Pth.FullName
$ExtraPaths = @("..", "..\..\..")
foreach ($ExtraPath in $ExtraPaths) {
    if ($PthText -notcontains $ExtraPath) {
        $PthText = @($PthText[0]) + @($ExtraPath) + @($PthText[1..($PthText.Count - 1)])
    }
}
$PthText = $PthText | ForEach-Object {
    if ($_ -eq "#import site") { "import site" } else { $_ }
}
Set-Content -LiteralPath $Pth.FullName -Value $PthText -Encoding ASCII

$PythonExe = Join-Path $OutDir "python.exe"
$GetPip = Join-Path $CacheDir "get-pip.py"
if (-not (Test-Path $GetPip)) {
    Write-Host "Downloading get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip
}

Write-Host "Installing pip into embedded Python"
& $PythonExe $GetPip --no-warn-script-location

Write-Host "Installing Fast runtime dependencies"
& $PythonExe -m pip install `
    --no-warn-script-location `
    --disable-pip-version-check `
    -i "https://pypi.tuna.tsinghua.edu.cn/simple" `
    -r $Requirements

Write-Host "Verifying embedded runtime"
$VerifyScript = Join-Path $CacheDir "verify_embedded_runtime.py"
@'
import importlib
mods = ["flask", "PIL", "numpy", "cv2", "rawpy", "imagehash", "pillow_heif", "piexif"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("\n".join(missing))
print("embedded runtime ok")
'@ | Set-Content -LiteralPath $VerifyScript -Encoding UTF8
& $PythonExe $VerifyScript

Write-Host "Done. Bundled Python is ready at $OutDir"
