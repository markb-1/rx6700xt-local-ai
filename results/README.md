# results/

Raw benchmark output, one subfolder per tool. Nothing in here is edited by hand.

## Layout

```
results/
  summary.csv              one row per (tool, backend, model, workload) headline number
  llamacpp/                raw llama-bench CSV, one file per sweep
  <tool>/                  raw output from other harness runners
```

File names: `YYYY-MM-DD-<model>-<what>.csv`.

## summary.csv schema

| Column | Meaning |
|--------|---------|
| `date` | ISO date of the run |
| `tool` | `llama.cpp`, `whisper.cpp`, `ollama`, ... |
| `tool_version` | build number, tag or commit |
| `backend` | `vulkan`, `rocm`, `directml`, `zluda`, `cpu` |
| `os` | `win11`, `ubuntu-24.04`, ... |
| `gpu_driver` | driver version string from `docs/hardware.md` |
| `model` | model name as published, e.g. `Qwen3-30B-A3B` |
| `quant` | `Q4_K_M`, `fp16`, ... |
| `config` | the knobs that matter, e.g. `ngl=99 ncmoe=24` |
| `workload` | `pp512`, `tg128`, `whisper-30s`, `sd-512x512-20steps` |
| `metric` | `tok/s`, `s`, `img/min` |
| `value` | number |
| `vram_mb` | peak VRAM if measured, else blank |
| `status` | `ok`, `fail`, `oom` |
| `notes` | error text or anything a reader needs |

## llama-bench CSV

Each raw file is llama-bench `-o csv` output with two extra leading columns added by `scripts/llamacpp/sweep.ps1`: `label` (the config name) and `start` (local timestamp). The rest are llama-bench's own columns. The two that matter most are `n_prompt`/`n_gen` (which test the row is) and `avg_ts` (tokens per second).

## PyTorch CSV

`scripts/pytorch/bench.py` writes one row per test: `start`, `torch` (version string with the `+rocm` suffix), `hip` (runtime version), `device`, `test`, `metric`, `value`, `status` (`ok`, `fail`, `skip`) and `notes`. A failed test is still a row, with the exception text in `notes`.

## ComfyUI CSV

`scripts/comfyui/bench.py` writes one row per run: `label`, `start`, `comfyui` and `torch` versions, `device`, `server_args`, `width`, `height`, `steps`, `sampler`, `rep`, `steps_per_s` (parsed from the sampler's tqdm line in the server log), `sampling_s` (KSampler start to VAEDecode start, over the websocket), `decode_s`, `total_s`. The server log for each invocation sits next to the CSV.
