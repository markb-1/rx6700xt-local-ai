<#
.SYNOPSIS
  Runs llama-bench once per configuration and appends every row to one CSV.

.DESCRIPTION
  Each config runs in its own llama-bench process, so an out-of-memory config
  fails on its own without aborting the sweep. Two columns are prepended to
  llama-bench's CSV output: `label` (your config name) and `start` (local time).

  Feed the output to summarize.ps1 to get a markdown table and summary.csv rows.

.EXAMPLE
  .\sweep.ps1 -Model ..\..\models\Qwen3-30B-A3B-Q4_K_M.gguf

.EXAMPLE
  # A quick two-point check with a different build
  .\sweep.ps1 -Bin ..\..\bin\llama.cpp\b11200-vulkan -Reps 1 -Configs @(
      @{ label = 'ngl0';  args = @('-ngl', '0') },
      @{ label = 'ngl99'; args = @('-ngl', '99') })
#>
param(
    [string]$Model = "$PSScriptRoot\..\..\models\Qwen3-30B-A3B-Q4_K_M.gguf",
    [string]$Bin = "$PSScriptRoot\..\..\bin\llama.cpp\b11146-vulkan",
    [string]$Out = "$PSScriptRoot\..\..\results\llamacpp\$(Get-Date -Format yyyy-MM-dd)-sweep-$(Get-Date -Format HHmm).csv",
    [int]$Reps = 3,
    [int]$Prompt = 512,
    [int]$Gen = 128,
    # Each entry is a label plus the extra llama-bench args for that config.
    # Default set is for a MoE model that does not fit in 12 GB: a layer sweep
    # (-ngl) and an expert-offload sweep (-ncmoe, all layers on GPU but N
    # layers' experts kept on CPU).
    [object[]]$Configs = @(
        @{ label = 'cpu-only';  args = @('-ngl', '0', '-nopo', '1') },
        @{ label = 'ngl0';      args = @('-ngl', '0') },
        @{ label = 'ngl8';      args = @('-ngl', '8') },
        @{ label = 'ngl16';     args = @('-ngl', '16') },
        @{ label = 'ngl20';     args = @('-ngl', '20') },
        @{ label = 'ngl24';     args = @('-ngl', '24') },
        @{ label = 'ngl28';     args = @('-ngl', '28') },
        @{ label = 'ngl30';     args = @('-ngl', '30') },
        @{ label = 'ncmoe48';   args = @('-ngl', '99', '-ncmoe', '48') },
        @{ label = 'ncmoe40';   args = @('-ngl', '99', '-ncmoe', '40') },
        @{ label = 'ncmoe32';   args = @('-ngl', '99', '-ncmoe', '32') },
        @{ label = 'ncmoe24';   args = @('-ngl', '99', '-ncmoe', '24') },
        @{ label = 'ncmoe20';   args = @('-ngl', '99', '-ncmoe', '20') },
        @{ label = 'ncmoe18';   args = @('-ngl', '99', '-ncmoe', '18') }
    )
)

$bench = Join-Path $Bin 'llama-bench.exe'
if (-not (Test-Path $bench)) { throw "llama-bench.exe not found at $bench. See docs/setup/llama-cpp-vulkan.md" }
if (-not (Test-Path $Model)) { throw "Model not found at $Model. See models/README.md" }

New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$header = $null
foreach ($c in $Configs) {
    Write-Host "== $($c.label): $($c.args -join ' ')"
    $start = Get-Date
    $rows = & $bench -m $Model -p $Prompt -n $Gen -r $Reps -o csv @($c.args) 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $rows) {
        Write-Host "   FAILED (exit $LASTEXITCODE)"
        continue
    }
    $rows = @($rows)
    if (-not $header) {
        $header = 'label,start,' + $rows[0]
        if (-not (Test-Path $Out)) { $header | Out-File -Encoding utf8 $Out }
    }
    foreach ($r in $rows[1..($rows.Count - 1)]) {
        "$($c.label),$($start.ToString('s')),$r" | Out-File -Encoding utf8 -Append $Out
    }
    Write-Host "   ok ($([int]((Get-Date) - $start).TotalSeconds)s)"
}
Write-Host "Results: $Out"
