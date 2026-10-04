<#
.SYNOPSIS
  Benchmarks whisper.cpp transcription across models and writes one CSV.

.DESCRIPTION
  Runs whisper-cli.exe once per (model, rep) on one audio file, parses the
  whisper_print_timings block, and appends a row per run to -Out. With -Summary
  it also appends mean rows to the cross-tool results/summary.csv.

  Pass -Backend to label the build (cpu, vulkan, ...). The default build in this
  repo is CPU because whisper.cpp ships no Windows Vulkan binary.

.EXAMPLE
  .\bench.ps1 -Models ..\..\models\ggml-base.en.bin, ..\..\models\ggml-large-v3.bin

.EXAMPLE
  .\bench.ps1 -Bin ..\..\bin\whisper.cpp\my-vulkan-build -Backend vulkan -Reps 1
#>
param(
    [string[]]$Models = @("$PSScriptRoot\..\..\models\ggml-base.en.bin", "$PSScriptRoot\..\..\models\ggml-large-v3.bin"),
    [string]$Audio = "$PSScriptRoot\..\..\models\samples\jfk.wav",
    [string]$Bin = "$PSScriptRoot\..\..\bin\whisper.cpp\b5130-cpu\Release",
    [string]$Backend = 'cpu',
    [string]$Version = 'b5130',
    [string]$Out = "$PSScriptRoot\..\..\results\whispercpp\$(Get-Date -Format yyyy-MM-dd)-bench-$(Get-Date -Format HHmm).csv",
    [string]$Summary,
    [string]$Os = 'win11',
    [string]$GpuDriver = '',
    [int]$Reps = 3,
    [int]$Threads = 8
)

$exe = Join-Path $Bin 'whisper-cli.exe'
if (-not (Test-Path $exe)) { throw "whisper-cli.exe not found at $exe. See docs/setup/whisper-cpp.md" }
if (-not (Test-Path $Audio)) { throw "Audio not found at $Audio" }
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null

$audioName = [IO.Path]::GetFileNameWithoutExtension($Audio)
$rows = @()

function Get-Ms([string[]]$lines, [string]$name) {
    $m = $lines | Select-String -Pattern "$name time\s*=\s*([\d.]+) ms" | Select-Object -Last 1
    if ($m) { return [double]$m.Matches[0].Groups[1].Value }
    return $null
}

foreach ($model in $Models) {
    if (-not (Test-Path $model)) { Write-Host "SKIP missing model $model"; continue }
    $label = ([IO.Path]::GetFileNameWithoutExtension($model)) -replace '^ggml-', ''
    for ($rep = 1; $rep -le $Reps; $rep++) {
        Write-Host "== $label rep $rep"
        $start = Get-Date
        # No -np: it suppresses the whisper_print_timings block we parse.
        $log = & $exe -m $model -f $Audio -t $Threads 2>&1 | ForEach-Object { "$_" }
        $exit = $LASTEXITCODE
        $wall = ((Get-Date) - $start).TotalSeconds
        $text = ($log | Where-Object { $_ -match '^\[\d\d:\d\d' } | ForEach-Object { ($_ -split '\]\s+', 2)[1] }) -join ' '

        $row = [pscustomobject]@{
            label = $label; start = $start.ToString('s'); build = $Version; backend = $Backend
            model = $label; audio = $audioName; threads = $Threads; rep = $rep
            load_ms = Get-Ms $log 'load'
            encode_ms = Get-Ms $log 'encode'
            decode_ms = Get-Ms $log 'decode'
            batchd_ms = Get-Ms $log 'batchd'
            total_ms = Get-Ms $log 'total'
            wall_s = [math]::Round($wall, 2)
            exit_code = $exit
            transcript = $text
        }
        $rows += $row
        if ($exit -eq 0 -and $row.total_ms) { Write-Host "   ok total $($row.total_ms) ms, encode $($row.encode_ms) ms" }
        else { Write-Host "   FAILED exit $exit"; $log | Select-Object -Last 5 | ForEach-Object { Write-Host "   $_" } }
    }
}

if (Test-Path $Out) { $rows | Export-Csv -NoTypeInformation -Append -Encoding utf8 $Out }
else { $rows | Export-Csv -NoTypeInformation -Encoding utf8 $Out }
Write-Host "Results: $Out"

if ($Summary) {
    $summaryRows = @()
    foreach ($g in ($rows | Where-Object { $_.exit_code -eq 0 -and $_.total_ms } | Group-Object label)) {
        $f = $g.Group[0]
        foreach ($pair in @(@('total', 'total_ms'), @('encode', 'encode_ms'))) {
            $mean = ($g.Group | Measure-Object $pair[1] -Average).Average
            $summaryRows += [pscustomobject]@{
                date = ([datetime]$f.start).ToString('yyyy-MM-dd'); tool = 'whisper.cpp'; tool_version = $Version
                backend = $Backend; os = $Os; gpu_driver = $GpuDriver
                model = $f.label; quant = 'f16'; config = "threads=$Threads"
                workload = "$audioName-$($pair[0])"; metric = 's'
                value = [math]::Round($mean / 1000, 3); vram_mb = ''; status = 'ok'; notes = "mean of $($g.Count) runs"
            }
        }
    }
    if (Test-Path $Summary) { $summaryRows | Export-Csv -NoTypeInformation -Append -Encoding utf8 $Summary }
    else { $summaryRows | Export-Csv -NoTypeInformation -Encoding utf8 $Summary }
    Write-Host "Appended $($summaryRows.Count) rows to $Summary"
}
