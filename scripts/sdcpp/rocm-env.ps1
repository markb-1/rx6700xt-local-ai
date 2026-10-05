<#
.SYNOPSIS
  Puts the TheRock ROCm wheel runtime on PATH for the current PowerShell session.

.DESCRIPTION
  Dot-source this before running the stable-diffusion.cpp ROCm build. It prepends the
  two wheel bin folders (HIP runtime and the BLAS libraries) so amdhip64_7.dll,
  hipblas.dll and rocblas.dll resolve. See docs/setup/stable-diffusion-cpp-rocm.md.

.EXAMPLE
  . .\scripts\sdcpp\rocm-env.ps1
  . .\scripts\sdcpp\rocm-env.ps1 -Venv C:\somewhere\else\.venv
#>
param(
    [string]$Venv = "$PSScriptRoot\..\..\bin\therock-7.14.0\.venv"
)

$site = Join-Path $Venv 'Lib\site-packages'
$core = Join-Path $site '_rocm_sdk_core\bin'
$libs = Join-Path $site '_rocm_sdk_libraries\bin'

foreach ($d in @($core, $libs)) {
    if (-not (Test-Path (Join-Path $d '*.dll'))) { throw "No DLLs in $d. Install the rocm wheels first; see docs/setup/stable-diffusion-cpp-rocm.md" }
}
if (-not (Test-Path (Join-Path $libs 'rocblas\library\*.dat'))) {
    Write-Warning "No rocBLAS kernel files under $libs\rocblas\library. Install rocm-sdk-device-<gfx> or matmuls will abort."
}

$env:PATH = "$((Resolve-Path $core).Path);$((Resolve-Path $libs).Path);$env:PATH"
Write-Host "ROCm runtime on PATH from $Venv"
