<#
.SYNOPSIS
  Turns a raw llama-bench sweep CSV into a markdown table and summary.csv rows.

.DESCRIPTION
  Groups rows by `label`, picks the prompt-processing (n_prompt > 0) and
  text-generation (n_gen > 0) results, and prints a markdown table to stdout.
  With -Summary it also appends one row per (label, workload) to the
  cross-tool summary file described in results/README.md.

.EXAMPLE
  .\summarize.ps1 ..\..\results\llamacpp\2026-09-30-qwen3-30b-a3b-ngl-sweep.csv

.EXAMPLE
  .\summarize.ps1 sweep.csv -Summary ..\..\results\summary.csv -Os win11 -GpuDriver 32.0.21045.5002
#>
param(
    [Parameter(Mandatory, Position = 0)][string]$Path,
    [string]$Summary,
    [string]$Os = 'win11',
    [string]$GpuDriver = '',
    [string]$Status = 'ok',
    [string]$Notes = ''
)

$rows = Import-Csv $Path
if (-not $rows) { throw "No rows in $Path" }

# Model name and quant from the file name, e.g. "Qwen3-30B-A3B-Q4_K_M"
$first = $rows[0]
$modelFile = [IO.Path]::GetFileNameWithoutExtension($first.model_filename)
$quant = ''
if ($modelFile -match '-((?:I?Q\d[_A-Za-z0-9]*)|F16|BF16|F32)$') { $quant = $Matches[1] }
$model = $modelFile
if ($quant) { $model = $modelFile.Substring(0, $modelFile.Length - $quant.Length - 1) }
$backend = $first.backends.ToLower()
$date = ([datetime]$first.start).ToString('yyyy-MM-dd')

$summaryRows = @()
$table = @()
$table += '| Config | ngl | ncmoe | Prompt tok/s | Gen tok/s |'
$table += '|--------|----:|------:|-------------:|----------:|'

foreach ($g in ($rows | Group-Object label)) {
    $pp = $g.Group | Where-Object { [int]$_.n_prompt -gt 0 } | Select-Object -First 1
    $tg = $g.Group | Where-Object { [int]$_.n_gen -gt 0 } | Select-Object -First 1
    $any = $pp
    if (-not $any) { $any = $tg }
    $config = "ngl=$($any.n_gpu_layers)"
    if ([int]$any.n_cpu_moe -gt 0) { $config += " ncmoe=$($any.n_cpu_moe)" }
    if ($any.no_op_offload -eq '1') { $config += ' nopo=1' }

    $ppVal = ''
    if ($pp) { $ppVal = [math]::Round([double]$pp.avg_ts, 1) }
    $tgVal = ''
    if ($tg) { $tgVal = [math]::Round([double]$tg.avg_ts, 1) }
    $table += "| $($g.Name) | $($any.n_gpu_layers) | $($any.n_cpu_moe) | $ppVal | $tgVal |"

    $pairs = @()
    if ($pp) { $pairs += ,@($pp, "pp$($pp.n_prompt)") }
    if ($tg) { $pairs += ,@($tg, "tg$($tg.n_gen)") }
    foreach ($pair in $pairs) {
        $r = $pair[0]
        $workload = $pair[1]
        $summaryRows += [pscustomobject]@{
            date = $date; tool = 'llama.cpp'; tool_version = "b$($r.build_number)"
            backend = $backend; os = $Os; gpu_driver = $GpuDriver
            model = $model; quant = $quant; config = $config; workload = $workload
            metric = 'tok/s'; value = [math]::Round([double]$r.avg_ts, 2)
            vram_mb = ''; status = $Status; notes = $Notes
        }
    }
}

"**$model $quant**, llama.cpp b$($first.build_number) $backend, $date"
''
$table -join "`n"

if ($Summary) {
    if (Test-Path $Summary) {
        $summaryRows | Export-Csv -NoTypeInformation -Append -Encoding utf8 $Summary
    } else {
        New-Item -ItemType Directory -Force (Split-Path $Summary) | Out-Null
        $summaryRows | Export-Csv -NoTypeInformation -Encoding utf8 $Summary
    }
    ''
    "Appended $($summaryRows.Count) rows to $Summary"
}
