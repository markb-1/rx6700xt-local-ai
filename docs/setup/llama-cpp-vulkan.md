# llama.cpp with the Vulkan backend

Status: **works**. This is the baseline path for the RX 6700 XT on Windows and needs no driver tricks.

## Install

1. Download the Windows Vulkan build from the llama.cpp releases page. The asset is named like `llama-b11146-bin-win-vulkan-x64.zip`.
2. Unzip into `bin/llama.cpp/b11146-vulkan/` (any folder works; pass it with `-Bin`).
3. Check the GPU is seen:

```powershell
.\bin\llama.cpp\b11146-vulkan\llama-bench.exe --list-devices
```

You should see `Vulkan0: AMD Radeon RX 6700 XT (12288 MiB)`.

## Flags that matter on this card

| Flag | What it does | Notes |
|------|--------------|-------|
| `-ngl N` | Number of transformer layers on the GPU | 12 GB fits an 8B or 14B model at Q4 fully; 30B needs a split |
| `-ncmoe N` | Keep the MoE experts of the first N layers on the CPU | MoE models only. All attention runs on GPU while experts spill to RAM |
| `-fa on` | Flash attention | Works on Vulkan; small win on long prompts |
| `-t 8` | CPU threads | Physical core count, not thread count, was fastest on the 5800X |

RDNA2 has no `VK_KHR_cooperative_matrix`, so llama.cpp uses scalar and fp16 matmul shaders. Prompt processing is where this shows up. Generation speed is bandwidth-bound and competitive.

## Running the sweep

```powershell
cd scripts\llamacpp
.\sweep.ps1 -Model ..\..\models\Qwen3-30B-A3B-Q4_K_M.gguf
.\summarize.ps1 ..\..\results\llamacpp\<file>.csv -Summary ..\..\results\summary.csv -GpuDriver 32.0.21045.5002
```

## Known issues

- None at b11146 with driver 32.0.21045.5002.
