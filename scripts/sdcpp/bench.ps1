<#
.SYNOPSIS
  Benchmarks stable-diffusion.cpp image generation across configs and writes one CSV.

.DESCRIPTION
  Runs sd-cli.exe once per (config, rep) with a fixed prompt and seed, parses the
  verbose log for stage timings, and appends a row per run to -Out. With -Summary
  it also appends mean rows to the cross-tool results/summary.csv.

  Parsed per run: text-encoder time, sampling time, VAE decode time, wall time,
  params VRAM, UNet and VAE compute buffers split by VRAM and RAM. A non-empty
  *_cpu_buffer_mb means ggml placed part of that graph on the CPU, usually because a
  tensor exceeded the Vulkan 2 GiB max buffer size (see docs/setup/stable-diffusion-cpp-vulkan.md).
  Steps per second = steps / sampling time.

.EXAMPLE
  .\bench.ps1 -Model ..\..\models\v1-5-pruned-emaonly-fp16.safetensors

.EXAMPLE
  .\bench.ps1 -Reps 1 -Configs @(@{ label = '768-euler_a-20'; width = 768; height = 768; steps = 20; sampler = 'euler_a'; args = @() })
#>
param(
    [string]$Model = "$PSScriptRoot\..\..\models\v1-5-pruned-emaonly-fp16.safetensors",
    [string]$Bin = "$PSScriptRoot\..\..\bin\stable-diffusion.cpp\master-929-vulkan",
    [string]$Out = "$PSScriptRoot\..\..\results\sdcpp\$(Get-Date -Format yyyy-MM-dd)-bench-$(Get-Date -Format HHmm).csv",
    [string]$Images = "$PSScriptRoot\..\..\results\sdcpp\images",
    [string]$Summary,
    [string]$Os = 'win11',
    [string]$GpuDriver = '',
    [int]$Reps = 3,
    [int]$Threads = 8,
    [string]$Prompt = 'a lighthouse on a rocky coast at sunset, oil painting',
    [string]$Negative = 'blurry, low quality',
    [int]$Seed = 42,
    [object[]]$Configs = @(
        @{ label = '512-euler_a-20';     width = 512; height = 512; steps = 20; sampler = 'euler_a'; args = @() },
        @{ label = '512-euler_a-20-fa';  width = 512; height = 512; steps = 20; sampler = 'euler_a'; args = @('--diffusion-fa') },
        @{ label = '512-dpmpp2m-20';     width = 512; height = 512; steps = 20; sampler = 'dpm++2m'; args = @() },
        @{ label = '768-euler_a-20';     width = 768; height = 768; steps = 20; sampler = 'euler_a'; args = @() },
        @{ label = '768-euler_a-20-fa-vaedirect'; width = 768; height = 768; steps = 20; sampler = 'euler_a'; args = @('--diffusion-fa', '--vae-conv-direct') }
    )
)

$exe = Join-Path $Bin 'sd-cli.exe'
if (-not (Test-Path $exe)) { throw "sd-cli.exe not found at $exe. See docs/setup/stable-diffusion-cpp-vulkan.md" }
if (-not (Test-Path $Model)) { throw "Model not found at $Model. See models/README.md" }
New-Item -ItemType Directory -Force (Split-Path $Out), $Images | Out-Null

$build = ''
$backend = ''
$modelName = [IO.Path]::GetFileNameWithoutExtension($Model)
$rows = @()

function Get-Num([string[]]$lines, [string]$pattern) {
    $m = $lines | Select-String -Pattern $pattern | Select-Object -Last 1
    if ($m) { return [double]$m.Matches[0].Groups[1].Value }
    return $null
}

foreach ($c in $Configs) {
    for ($rep = 1; $rep -le $Reps; $rep++) {
        Write-Host "== $($c.label) rep $rep"
        $start = Get-Date
        $img = Join-Path $Images "$($c.label)-r$rep.png"
        $argList = @('-m', $Model, '-p', $Prompt, '-n', $Negative,
                     '-W', $c.width, '-H', $c.height, '--steps', $c.steps,
                     '--sampling-method', $c.sampler, '-s', $Seed, '-t', $Threads,
                     '-o', $img, '-v') + @($c.args)
        $log = & $exe @argList 2>&1 | ForEach-Object { "$_" }
        $exit = $LASTEXITCODE
        $wall = ((Get-Date) - $start).TotalSeconds

        if (-not $build) {
            $m = $log | Select-String 'commit ([0-9a-f]+)' | Select-Object -First 1
            if ($m) { $build = $m.Matches[0].Groups[1].Value }
            # The HIP build logs through the CUDA code path ("ggml_cuda_init: found 1 ROCm devices"), so test ROCm first
            if ($log -match 'ggml_vulkan: Found') { $backend = 'vulkan' }
            elseif ($log -match 'found \d+ ROCm devices') { $backend = 'rocm' }
            elseif ($log -match 'found \d+ CUDA devices') { $backend = 'cuda' }
            else { $backend = 'cpu' }
        }

        $sampling = Get-Num $log 'sampling completed, taking ([\d.]+)s'
        $row = [pscustomobject]@{
            label = $c.label; start = $start.ToString('s'); build = $build; backend = $backend
            model = $modelName; width = $c.width; height = $c.height; steps = $c.steps
            sampler = $c.sampler; extra_args = ($c.args -join ' '); rep = $rep
            cond_s = Get-Num $log 'get_learned_condition completed, taking ([\d.]+)s'
            sampling_s = $sampling
            decode_s = Get-Num $log 'decode_first_stage completed, taking ([\d.]+)s'
            wall_s = [math]::Round($wall, 2)
            steps_per_s = if ($sampling) { [math]::Round($c.steps / $sampling, 3) } else { $null }
            params_mb = Get-Num $log 'total params memory size = ([\d.]+)MB'
            unet_buffer_mb = Get-Num $log 'unet compute buffer size: ([\d.]+) MB\(VRAM\)'
            unet_cpu_buffer_mb = Get-Num $log 'unet compute buffer size: ([\d.]+) MB\(RAM\)'
            vae_buffer_mb = Get-Num $log 'vae compute buffer size: ([\d.]+) MB\(VRAM\)'
            vae_cpu_buffer_mb = Get-Num $log 'vae compute buffer size: ([\d.]+) MB\(RAM\)'
            exit_code = $exit
        }
        $rows += $row
        $status = if ($exit -eq 0 -and $sampling) { "ok $($row.steps_per_s) steps/s, sampling $sampling s" } else { "FAILED exit $exit" }
        if ($row.unet_cpu_buffer_mb -or $row.vae_cpu_buffer_mb) { $status += " (CPU FALLBACK: unet $($row.unet_cpu_buffer_mb) MB, vae $($row.vae_cpu_buffer_mb) MB in RAM)" }
        Write-Host "   $status"
        if ($exit -ne 0) { $log | Select-Object -Last 5 | ForEach-Object { Write-Host "   $_" } }
    }
}

if (Test-Path $Out) { $rows | Export-Csv -NoTypeInformation -Append -Encoding utf8 $Out }
else { $rows | Export-Csv -NoTypeInformation -Encoding utf8 $Out }
Write-Host "Results: $Out"

if ($Summary) {
    $summaryRows = @()
    foreach ($g in ($rows | Where-Object { $_.exit_code -eq 0 -and $_.sampling_s } | Group-Object label)) {
        $f = $g.Group[0]
        $mean = ($g.Group | Measure-Object steps_per_s -Average).Average
        $summaryRows += [pscustomobject]@{
            date = ([datetime]$f.start).ToString('yyyy-MM-dd'); tool = 'stable-diffusion.cpp'; tool_version = $f.build
            backend = $f.backend; os = $Os; gpu_driver = $GpuDriver
            model = 'SD1.5'; quant = 'fp16'; config = "$($f.sampler) $($f.extra_args)".Trim()
            workload = "sd-$($f.width)x$($f.height)-$($f.steps)steps"; metric = 'steps/s'
            value = [math]::Round($mean, 2); vram_mb = ''; status = 'ok'; notes = "mean of $($g.Count) runs"
        }
    }
    if (Test-Path $Summary) { $summaryRows | Export-Csv -NoTypeInformation -Append -Encoding utf8 $Summary }
    else { $summaryRows | Export-Csv -NoTypeInformation -Encoding utf8 $Summary }
    Write-Host "Appended $($summaryRows.Count) rows to $Summary"
}
