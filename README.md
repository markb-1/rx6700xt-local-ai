# rx6700xt-local-ai

What local AI tools actually run on an AMD Radeon RX 6700 XT (RDNA2, `gfx1031`), with exact versions, the flags that work, reproducible benchmarks, and the workarounds for everything that assumes ROCm or CUDA.

The card has 12 GB of VRAM and plenty of bandwidth, but AMD lists it as **runtime only** for ROCm on Windows, with no HIP SDK support. Most AI tooling therefore either falls back to Vulkan, needs community-patched libraries, or does not run. This repo records which is which, so you do not have to find out by trial and error.

Everything here was measured on one machine, described in [docs/hardware.md](docs/hardware.md). If you have a different RDNA2 card (6600 to 6950 XT), the compatibility column almost certainly applies to you too, the numbers less so.

## Compatibility matrix

Status key: **works** means tested here and usable. **works\*** means it needs a documented workaround. **untested** means on the list. **fails** means tested and broken, with the error recorded.

| Tool | Backend | OS | Status | Version tested | Notes |
|------|---------|----|--------|----------------|-------|
| llama.cpp | Vulkan | Windows 11 | **works** | b11146 | Baseline path. [Setup](docs/setup/llama-cpp-vulkan.md), results below |
| stable-diffusion.cpp | Vulkan | Windows 11 | **works** | master-929 | SD 1.5 at 512x512 in 11 s. 768x768 is 30x slower, see results. [Setup](docs/setup/stable-diffusion-cpp-vulkan.md) |
| stable-diffusion.cpp | ROCm 7.14 | Windows 11 | fails | master-929 | `0xC0000135`, needs `amdhip64_7.dll` from the HIP SDK 7. The DLL does bundle `gfx1031` kernels, so retest after installing the SDK |
| whisper.cpp | CPU | Windows 11 | **works** | b5130 | Baseline for the GPU build. [Setup](docs/setup/whisper-cpp.md) |
| whisper.cpp | Vulkan | Windows 11 | untested | | No prebuilt Windows Vulkan binary in any release. Needs a source build with the Vulkan SDK |
| LM Studio | Vulkan | Windows 11 | untested | | Same llama.cpp backend under a GUI |
| koboldcpp | Vulkan | Windows 11 | untested | | |
| Ollama | ROCm | Windows 11 | fails | 0.6.7 | Stock build falls back to CPU, forced ROCm fails with `RMS_NORM failed`. Retest with patched rocBLAS pending |
| Ollama | ROCm + patched rocBLAS | Windows 11 | untested | | Community `gfx1031` libraries |
| koboldcpp-rocm | ROCm + patched rocBLAS | Windows 11 | untested | | |
| ComfyUI | DirectML | Windows 11 | untested | | torch-directml |
| ComfyUI | ZLUDA | Windows 11 | untested | | Needs patched rocBLAS too |
| PyTorch | DirectML | Windows 11 | untested | | |
| PyTorch | ROCm | Windows 11 | fails | | No HIP SDK for `gfx1031`. Not fixable without rebuilding ROCm |
| llama.cpp | ROCm (HIP) | Linux | untested | | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| PyTorch | ROCm | Linux | untested | | Same override. Only route to real training on this card |
| ComfyUI | ROCm | Linux | untested | | |
| vLLM | ROCm | Linux | untested | | Expected to be hard |

## Headline results

### llama.cpp b11146, Vulkan, Qwen3-30B-A3B Q4_K_M

A 30B mixture-of-experts model that does not fit in 12 GB. Two ways to split it: put the first N layers on the GPU (`-ngl N`), or put every layer on the GPU but keep the experts of the first N layers in system RAM (`-ngl 99 -ncmoe N`). Prompt is 512 tokens, generation is 128 tokens, 3 repetitions each.

| Config | ngl | ncmoe | Prompt tok/s | Gen tok/s |
|--------|----:|------:|-------------:|----------:|
| cpu-only | 0 | 0 | 89.4 | 20.6 |
| ngl0 | 0 | 0 | 223.4 | 15.1 |
| ngl8 | 8 | 0 | 247.0 | 20.3 |
| ngl16 | 16 | 0 | 285.5 | 23.9 |
| ngl20 | 20 | 0 | 310.2 | 26.5 |
| ngl24 | 24 | 0 | 335.1 | 28.9 |
| ngl28 | 28 | 0 | 365.6 | 32.5 |
| ngl30 | 30 | 0 | 384.2 | 34.1 |
| ncmoe48 | 99 | 48 | 237.9 | 18.5 |
| ncmoe40 | 99 | 40 | 265.6 | 21.1 |
| ncmoe32 | 99 | 32 | 308.1 | 24.9 |
| ncmoe24 | 99 | 24 | 361.1 | 28.5 |
| ncmoe20 | 99 | 20 | 393.2 | 31.4 |
| ncmoe18 | 99 | 18 | 350.0 | 33.0 |

Takeaways:

- `-ngl 30` was the most layers that fit. Generation reaches 34 tok/s, which is comfortable for chat.
- `cpu-only` (`-nopo 1`, no op offload) beats `ngl0` for generation because with `-ngl 0` llama.cpp still ships large matmuls to the GPU and the PCIe round trip costs more than it saves at batch size 1. The reverse holds for prompt processing.
- Expert offload (`-ncmoe`) gives the best prompt speed at `ncmoe20`, but the plain layer split wins slightly on generation at the memory limit.

Raw data: [results/llamacpp/2026-09-30-qwen3-30b-a3b-ngl-sweep.csv](results/llamacpp/2026-09-30-qwen3-30b-a3b-ngl-sweep.csv).

### stable-diffusion.cpp master-929, Vulkan, Stable Diffusion 1.5 fp16

Fixed prompt and seed, 3 runs per config, mean of the sampling stage only. Wall time adds about 1.5 s for text encoding and VAE decode at 512x512.

| Config | Steps/s | Sampling | VAE decode | UNet buffer |
|--------|--------:|---------:|-----------:|------------:|
| 512x512, euler_a, 20 steps | 2.08 | 9.6 s | 0.9 s | 560 MB |
| 512x512, euler_a, 20 steps, `--diffusion-fa` | 1.23 | 16.4 s | 0.9 s | 123 MB |
| 512x512, dpm++2m, 20 steps | 2.11 | 9.5 s | 0.9 s | 560 MB |
| 768x768, euler_a, 20 steps | 0.07 | 289.8 s | 6.5 s | 2626 MB |

Takeaways:

- 512x512 is comfortable: about 11 s per image end to end.
- Flash attention is a loss on this card. It cuts the UNet buffer from 560 MB to 123 MB but runs 40% slower, consistent with RDNA2 lacking the matrix-core path the Vulkan flash-attention shader is written for. Leave `--diffusion-fa` off unless VRAM is the constraint.
- 768x768 collapses to 290 s, around 30x slower for 2.25x the pixels, and VAE decode slows 7x as well. Total VRAM in use is under 5 GB by the reported buffers, so this is not simple VRAM exhaustion. Cause not yet identified. If you need larger images on this card today, generate at 512 and upscale.

Sample output: [results/sdcpp/images/sample-sd15-512-euler_a-20.png](results/sdcpp/images/sample-sd15-512-euler_a-20.png). Raw data: [results/sdcpp/2026-10-04-sd15-vulkan-bench.csv](results/sdcpp/2026-10-04-sd15-vulkan-bench.csv).

### whisper.cpp b5130, CPU, JFK clip (11 s of speech)

CPU baseline on the Ryzen 7 5800X with 8 threads, 3 runs each. This is the number a Vulkan build has to beat.

| Model | Encode | Total | Realtime factor |
|-------|-------:|------:|----------------:|
| base.en | 0.30 s | 0.68 s | 16x faster than realtime |
| large-v3 | 6.97 s | 11.6 s | about realtime |

Raw data: [results/whispercpp/2026-10-04-cpu-bench.csv](results/whispercpp/2026-10-04-cpu-bench.csv).

Cross-tool headline numbers for everything above: [results/summary.csv](results/summary.csv).

## Reproducing

```powershell
# 1. Put a llama.cpp Vulkan build in bin\llama.cpp\<build>\ and a GGUF in models\
# 2. Run the sweep (about 20 minutes for the default 14 configs)
cd scripts\llamacpp
.\sweep.ps1 -Model ..\..\models\Qwen3-30B-A3B-Q4_K_M.gguf -Bin ..\..\bin\llama.cpp\b11146-vulkan

# 3. Turn the raw CSV into a markdown table and summary.csv rows
.\summarize.ps1 ..\..\results\llamacpp\<file>.csv -Summary ..\..\results\summary.csv -GpuDriver <your driver>
```

Every runner writes raw output under `results/<tool>/` and headline rows to `results/summary.csv` using the schema in [results/README.md](results/README.md).

## Layout

```
docs/hardware.md        the test machine and why gfx1031 is awkward
docs/setup/<tool>.md    install steps, flags, known issues, one file per tool x backend
scripts/<tool>/         harness runner and summarizer for that tool
results/<tool>/         raw benchmark output, never hand-edited
results/summary.csv     one row per headline number across all tools
bin/, models/           git-ignored; binaries and model files live here locally
```

## Roadmap

1. Find out why stable-diffusion.cpp falls off a cliff at 768x768 on Vulkan, and test SDXL.
2. Install the AMD HIP SDK 7.x and retest the stable-diffusion.cpp ROCm build, which already bundles `gfx1031` kernels.
3. Build whisper.cpp with the Vulkan backend and compare against the CPU baseline.
4. Fill the remaining Windows Vulkan rows: LM Studio, koboldcpp.
5. Retest Ollama and koboldcpp-rocm with community-built `gfx1031` rocBLAS libraries, and document the exact files and versions.
6. ZLUDA and DirectML for ComfyUI.
7. Linux dual boot with the `10.3.0` override: llama.cpp HIP, PyTorch, ComfyUI.
8. Profile llama.cpp's Vulkan fallback matmul shaders on RDNA2 and report upstream.

## Contributing

Results from other RDNA2 cards are welcome. Open a pull request that adds your raw output under `results/<tool>/`, your headline rows to `results/summary.csv` with a `notes` entry naming the card, and a short hardware section to the setup doc if your path differed. Please include tool version and GPU driver version in every row.

## License

MIT. See [LICENSE](LICENSE).
