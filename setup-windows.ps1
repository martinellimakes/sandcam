<#
.SYNOPSIS
    Downloads and compiles all Windows dependencies for sandcam.

.DESCRIPTION
    1. Checks for required tools (Git, CMake, uv, MSVC)
    2. Clones OpenKinect/libfreenect (skips if already present)
    3. Builds libfreenect.dll using the bundled libusb binaries
    4. Installs to C:\libfreenect and copies runtime DLLs next to main.py
    5. Runs uv sync to install Python dependencies

.PARAMETER LibFreenectSrc
    Where to clone the libfreenect source tree.
    Defaults to a sibling directory of the project root.

.PARAMETER Force
    Re-run the CMake build and install even if the DLLs already exist.

.EXAMPLE
    .\setup-windows.ps1
    .\setup-windows.ps1 -LibFreenectSrc C:\dev\libfreenect
    .\setup-windows.ps1 -Force
#>

[CmdletBinding()]
param(
    [string] $LibFreenectSrc = (Join-Path (Split-Path $PSScriptRoot -Parent) "libfreenect"),
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot  = $PSScriptRoot
$InstallDir   = "C:\libfreenect"
$FreenectDll  = Join-Path $ProjectRoot "freenect.dll"
$LibusbDll    = Join-Path $ProjectRoot "libusb-1.0.dll"

# ─────────────────────────────────────────────────────────────────────────────
function Write-Step($msg)    { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK($msg)      { Write-Host "   ok  $msg" -ForegroundColor Green }
function Write-Skip($msg)    { Write-Host "   --  $msg" -ForegroundColor DarkGray }
function Fail($msg)          { Write-Host "`n[FAIL] $msg" -ForegroundColor Red; exit 1 }

function Require-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Fail "$name not found.  $hint"
    }
    Write-OK "$name"
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Checking prerequisites"

Require-Command git   "Install Git from https://git-scm.com/"
Require-Command cmake "Install CMake: winget install Kitware.CMake"
Require-Command uv    "Install uv: winget install astral-sh.uv  or  pip install uv"

# MSVC — CMake will find it automatically, but warn early if absent
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null
    if ($vsPath) {
        Write-OK "MSVC at $vsPath"
    } else {
        Write-Warning "MSVC C++ tools not found.  Open Visual Studio Installer and add 'Desktop development with C++'."
    }
} else {
    Write-Warning "Visual Studio Installer not found — CMake may fail.  Install VS Build Tools from https://visualstudio.microsoft.com/downloads/"
}

# ─────────────────────────────────────────────────────────────────────────────
# Short-circuit if DLLs already exist and -Force was not passed
if (-not $Force -and (Test-Path $FreenectDll) -and (Test-Path $LibusbDll)) {
    Write-Step "DLLs already present — skipping build"
    Write-Skip "Pass -Force to rebuild anyway"
    Write-Step "Syncing Python dependencies"
    Push-Location $ProjectRoot; uv sync; Pop-Location
    Write-Host "`n>> Setup already complete.  Run: uv run python main.py" -ForegroundColor Green
    exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Cloning libfreenect source"

if (Test-Path (Join-Path $LibFreenectSrc ".git")) {
    Write-Skip "Already cloned at $LibFreenectSrc"
} else {
    git clone https://github.com/OpenKinect/libfreenect $LibFreenectSrc
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed" }
    Write-OK "Cloned to $LibFreenectSrc"
}

# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Locating bundled libusb binaries"

# libfreenect ships pre-built libusb DLLs for several VS versions
$LibusbLib = $null
$LibusbBinDll = $null
foreach ($vs in @("VS2022", "VS2019", "VS2017", "VS2015")) {
    $candidate = Join-Path $LibFreenectSrc "libusb-1.0.29\$vs\MS64\dll\libusb-1.0.lib"
    if (Test-Path $candidate) {
        $LibusbLib    = $candidate
        $LibusbBinDll = Join-Path $LibFreenectSrc "libusb-1.0.29\$vs\MS64\dll\libusb-1.0.dll"
        Write-OK "Using bundled libusb for $vs"
        break
    }
}
if (-not $LibusbLib) {
    Fail "Bundled libusb binaries not found inside $LibFreenectSrc.  The clone may be incomplete."
}

$LibusbInc = Join-Path $LibFreenectSrc "libusb-1.0.29\include"

# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Configuring CMake"

$BuildDir = Join-Path $LibFreenectSrc "build"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir

cmake .. `
    "-DCMAKE_INSTALL_PREFIX=$InstallDir" `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_FAKENECT=OFF `
    -DBUILD_PYTHON3=OFF `
    -DBUILD_C_SYNC=OFF `
    "-DLIBUSB_1_INCLUDE_DIR=$LibusbInc" `
    "-DLIBUSB_1_LIBRARY=$LibusbLib"

if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "CMake configure failed" }

# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Building libfreenect"

cmake --build . --config Release
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "Build failed" }

Pop-Location

# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Installing to $InstallDir"

Push-Location $BuildDir
cmake --install . --config Release
$installExitCode = $LASTEXITCODE
Pop-Location

if ($installExitCode -ne 0) {
    Fail "Install to $InstallDir failed.`nIf you see a permissions error, re-run this script from an elevated (Administrator) PowerShell."
}
Write-OK "Installed"

# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Copying runtime DLLs to project root"

Copy-Item (Join-Path $InstallDir "lib\freenect.dll") -Destination $ProjectRoot -Force
Copy-Item $LibusbBinDll -Destination $ProjectRoot -Force
Write-OK "freenect.dll"
Write-OK "libusb-1.0.dll"

# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Installing Python dependencies"

Push-Location $ProjectRoot
uv sync
Pop-Location

# ─────────────────────────────────────────────────────────────────────────────
Write-Host @"

>> All done.

Next steps
----------
1. Install Zadig          https://zadig.akeo.ie/
   Plug in the Kinect, open Zadig, select the Kinect Camera device,
   replace its driver with WinUSB.

2. In main.py switch the source:
       # source = MouseSimulator(WIDTH, HEIGHT)
       source = KinectV1Source(WIDTH, HEIGHT)

3. Run:
       uv run python main.py

"@ -ForegroundColor Green
