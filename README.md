# rx6700xt-local-ai

What local AI tools actually run on an AMD Radeon RX 6700 XT (RDNA2, `gfx1031`), with exact versions, the flags that work, reproducible benchmarks, and the workarounds for everything that assumes ROCm or CUDA.

The card has 12 GB of VRAM and plenty of bandwidth, but AMD lists it as **runtime only** for ROCm on Windows, with no HIP SDK support. Most AI tooling therefore either falls back to Vulkan, needs community-patched libraries, or does not run. This repo records which is which, so you do not have to find out by trial and error.

Everything here was measured on one machine, described in [docs/hardware.md](docs/hardware.md). If you have a different RDNA2 card (6600 to 6950 XT), the compatibility column almost certainly applies to you too, the numbers less so.

## Compatibility matrix

Status key: **works** means tested here and usable. **works\*** means it needs a documented workaround. **untested** means on the list. **fails** means tested and broken, with the error recorded.

| Tool | Backend | OS | Status | Version tested | Notes |
|------|---------|----|--------|----------------|-------|
| llama.cpp | Vulkan | Windows 11 | **works** | b11146 | Baseline path. [Setup](docs/setup/llama-cpp-vulkan.md), results below |
| stable-diffusion.cpp | Vulkan | Windows 11 | **works\*** | master-929 | SD 1.5 at 512x512 in 11 s with no setup. At 768x768 and above the driver's 2 GiB buffer cap pushes attention onto the CPU, 30x slower; one env var or two flags fix it. [Setup](docs/setup/stable-diffusion-cpp-vulkan.md) |
| stable-diffusion.cpp | ROCm 7.14 (HIP) | Windows 11 | **works\*** | master-929 | Fails out of the box (`0xC0000135`, no HIP 7 runtime). Fixed by pip-installing AMD's ROCm 7.14.0 wheels plus the `gfx1031` kernel package; no SDK, no driver change. Then 27% faster than Vulkan at 512x512 and 11x faster at 768x768 as shipped. [Setup](docs/setup/stable-diffusion-cpp-rocm.md) |
| whisper.cpp | Vulkan | Windows 11 | **works\*** | b5130 | No prebuilt Windows Vulkan binary exists, so this is a source build. large-v3 inference 4x faster than CPU. [Setup](docs/setup/whisper-cpp.md) |
| whisper.cpp | CPU | Windows 11 | **works** | b5130 | Prebuilt zip. Baseline for the row above |
| LM Studio | Vulkan | Windows 11 | untested | | Same llama.cpp backend under a GUI |
| koboldcpp | Vulkan | Windows 11 | untested | | |
| Ollama | ROCm | Windows 11 | fails | 0.6.7 | Stock build falls back to CPU, forced ROCm fails with `RMS_NORM failed`. Retest with patched rocBLAS pending |
| Ollama | ROCm + patched rocBLAS | Windows 11 | untested | | Community `gfx1031` libraries |
| koboldcpp-rocm | ROCm + patched rocBLAS | Windows 11 | untested | | |
| ComfyUI | ROCm (HIP) | Windows 11 | **works\*** | 0.38.0 | On the PyTorch row's torch build. SD 1.5 at 7.0 steps/s at 512x512, 2x stable-diffusion.cpp. First run per resolution is slow while MIOpen tunes. [Setup](docs/setup/comfyui-rocm-windows.md) |
| ComfyUI | DirectML | Windows 11 | untested | | torch-directml. Moot now that ROCm works |
| ComfyUI | ZLUDA | Windows 11 | untested | | Needs patched rocBLAS too. Moot now that ROCm works |
| PyTorch | ROCm (HIP) | Windows 11 | **works\*** | 2.9.1+rocm7.13.0 | AMD's wheel index has Windows torch builds with `gfx1031` kernels. fp16 GEMM at 21 TFLOPS, training works. Versions must be pinned by hand. [Setup](docs/setup/pytorch-rocm-windows.md) |
| PyTorch | DirectML | Windows 11 | untested | | Probably moot now that ROCm works |
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
| 768x768, euler_a, 20 steps, as shipped | 0.07 | 289.8 s | 6.5 s | 284 MB VRAM + 2626 MB RAM |
| 768x768, euler_a, 20 steps, `--diffusion-fa --vae-conv-direct` | 0.47 | 43.0 s | 1.3 s | 277 MB |
| 768x768, euler_a, 20 steps, `GGML_VK_FORCE_MAX_BUFFER_SIZE=3G` | 0.79 | 25.3 s | 1.7 s | 5277 MB |
| 512x512, euler_a, 20 steps, `GGML_VK_FORCE_MAX_BUFFER_SIZE=3G` | 2.14 | 9.3 s | 0.9 s | 560 MB |

Takeaways:

- 512x512 is comfortable: about 11 s per image end to end.
- Flash attention is a loss on this card at 512x512. It cuts the UNet buffer from 560 MB to 123 MB but runs 40% slower, consistent with RDNA2 lacking the matrix-core path the Vulkan flash-attention shader is written for. Leave `--diffusion-fa` off at 512.
- 768x768 as shipped collapses to 290 s because the AMD Windows Vulkan driver caps single buffers at 2 GiB. The UNet's attention score matrix (9216 tokens squared, 8 heads, fp32, 2.6 GB) and the VAE's im2col tensor (2.7 GB) both cross that line, so stable-diffusion.cpp quietly runs those ops on the CPU. The 2626 MB "UNet buffer" in the shipped row is system RAM. Two fixes: `--diffusion-fa --vae-conv-direct` avoids building the big tensors (6.6x faster), or `GGML_VK_FORCE_MAX_BUFFER_SIZE=3221225472` with `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` set the same tells ggml to ignore the cap, which the driver accepts (10.7x faster, and no change at 512). Details and the log lines to look for are in the [setup doc](docs/setup/stable-diffusion-cpp-vulkan.md#the-2-gib-buffer-limit-768x768-and-above).

Sample output: [512x512](results/sdcpp/images/sample-sd15-512-euler_a-20.png) and [768x768 with the env var](results/sdcpp/images/sample-sd15-768-euler_a-20-forcemax3g.png). Raw data: [2026-10-04 bench](results/sdcpp/2026-10-04-sd15-vulkan-bench.csv) and [2026-10-05 768 fixes](results/sdcpp/2026-10-05-sd15-vulkan-768-fallback.csv).

### stable-diffusion.cpp master-929, ROCm (HIP) versus Vulkan, Stable Diffusion 1.5 fp16

Same binary release, same prompt, seed and settings. The ROCm build runs with AMD's ROCm 7.14.0 Python wheels as its runtime (see the [setup doc](docs/setup/stable-diffusion-cpp-rocm.md)); no HIP SDK and the Adrenalin driver untouched. 3 runs per config, mean of the sampling stage.

| Config | ROCm steps/s | Vulkan steps/s | ROCm sampling | ROCm VAE decode |
|--------|-------------:|---------------:|--------------:|----------------:|
| 512x512, euler_a, 20 steps | 2.64 | 2.08 | 7.6 s | 0.9 s |
| 512x512, euler_a, 20 steps, `--diffusion-fa` | 3.28 | 1.23 | 6.1 s | 0.9 s |
| 512x512, dpm++2m, 20 steps | 2.64 | 2.11 | 7.6 s | 0.9 s |
| 768x768, euler_a, 20 steps, as shipped | 0.77 | 0.07 | 26.1 s | 1.6 s |
| 768x768, euler_a, 20 steps, `--diffusion-fa` | 1.18 | not run | 17.0 s | 1.6 s |
| 768x768, euler_a, 20 steps, `--diffusion-fa --vae-conv-direct` | 1.18 | 0.47 | 17.0 s | 17.9 s |

Takeaways:

- ROCm is the faster backend for image generation on this card: 27% at 512x512 with default settings, and the best 768x768 result (20 s per image end to end with flash attention) is 2.2x the best Vulkan result.
- Flash attention flips sign between backends. On Vulkan it costs 40%; on HIP it gains 24% at 512 and 54% at 768. Use `--diffusion-fa` on ROCm, leave it off on Vulkan below 768.
- `--vae-conv-direct` is the Vulkan fix for 768x768 but a trap on ROCm: decode goes from 1.6 s to 17.9 s. On ROCm just leave the VAE flags off.
- HIP has no 2 GiB per-buffer cap, so 768x768 works as shipped with a 2.7 GB UNet buffer in VRAM.
- The out-of-the-box failure was never about the card. AMD's HIP SDK still lists RX 6000 as unsupported, but the wheels the binary was built against include `gfx1031` rocBLAS kernels. The same index also carries PyTorch wheels for `gfx1031` on Windows, which is the next row to test.

Sample output: [512x512 on ROCm](results/sdcpp/images/sample-sd15-512-euler_a-20-rocm.png). Raw data: [results/sdcpp/2026-10-05-sd15-rocm-bench.csv](results/sdcpp/2026-10-05-sd15-rocm-bench.csv).

### PyTorch 2.9.1 on ROCm 7.13.0, Windows

Installed from AMD's wheel index with the `gfx1031` device packages, no HIP SDK, no driver change. 10 reps per test. The [setup doc](docs/setup/pytorch-rocm-windows.md) has the pinned install command and the two version traps.

| Test | Result |
|------|-------:|
| 4096x4096 GEMM, fp16 | 21.1 TFLOPS (card peak 26.4) |
| 4096x4096 GEMM, fp32 | 11.5 TFLOPS (card peak 13.2) |
| 4096x4096 GEMM, bf16 | 6.0 TFLOPS (no bf16 hardware on RDNA2) |
| ResNet-18 forward, fp16, batch 8 | 2.6 ms |
| ResNet-18 train step, fp32, batch 8 | 21.0 ms |
| MLP train step, Adam, batch 256 | 2.9 ms |

Takeaways:

- This overturns the row that said PyTorch on ROCm was impossible on this card without Linux. rocBLAS reaches 80 to 87% of theoretical throughput, so these are tuned kernels, not fallbacks.
- Use fp16 for mixed precision, not bf16.
- MIOpen warns that its composable-kernel grouped-conv library is missing for `gfx1031`. Plain convolutions are fine; depthwise and grouped convs may be slower than on supported cards.

Raw data: [results/pytorch/2026-10-05-torch-bench.csv](results/pytorch/2026-10-05-torch-bench.csv).

### ComfyUI 0.38.0 on ROCm, Windows, Stable Diffusion 1.5 fp16

Same checkpoint, prompt, seed and step count as the stable-diffusion.cpp rows, driven headlessly through the API by `scripts/comfyui/bench.py`. 3 runs per config, steps/s from the sampler's progress counter, total from prompt submission to image saved.

| Config | Steps/s | Total per image | sd.cpp ROCm best | sd.cpp Vulkan best |
|--------|--------:|----------------:|-----------------:|-------------------:|
| 512x512, euler_ancestral, 20 steps | 7.0 | 4.4 s | 3.28 | 2.08 |
| 512x512, same, `--fp16-vae` | 7.0 | 4.2 s | | |
| 768x768, euler_ancestral, 20 steps | 2.27 | 11.2 s | 1.18 | 0.79 |
| 768x768, same, `--fp16-vae` | 2.27 | 10.3 s | | |

Takeaways:

- ComfyUI on PyTorch is 2x stable-diffusion.cpp on the same HIP runtime and the same weights. rocBLAS and MIOpen carry tuned kernels for this card; ggml's HIP kernels do not.
- Keep ComfyUI's default attention. `--use-pytorch-cross-attention`, the usual AMD advice, is 23% slower at 512 and 2x slower at 768 here. `--fp16-vae` is a free win for decode.
- The first generation at each new resolution takes minutes, nearly all in VAE decode, while MIOpen searches for kernels. It caches the result under `~/.miopen/` and never repeats it for that shape.

Samples: [512x512](results/comfyui/images/sample-sd15-512-euler_a-20-comfyui.png) and [768x768](results/comfyui/images/sample-sd15-768-euler_a-20-comfyui.png). Raw data: [results/comfyui/2026-10-05-comfyui-bench.csv](results/comfyui/2026-10-05-comfyui-bench.csv).

### whisper.cpp b5130, Vulkan versus CPU, JFK clip (11 s of speech)

Vulkan build compiled from source, CPU build from the prebuilt zip, 8 threads, 3 runs each. Vulkan numbers are the warm runs; the first Vulkan run of a session pays a one-off shader compile of about 2.5 s.

| Model | Backend | Load | Encode | Decode | Total |
|-------|---------|-----:|-------:|-------:|------:|
| base.en | CPU | 0.12 s | 0.30 s | 0.17 s | 0.68 s |
| base.en | Vulkan | 0.15 s | 0.11 s | 0.12 s | 0.46 s |
| large-v3 | CPU | 2.0 s | 6.97 s | 2.36 s | 11.6 s |
| large-v3 | Vulkan | 2.5 s | 1.28 s | 0.90 s | 4.84 s |

Takeaways:

- large-v3 goes from roughly realtime on the CPU to 2.3x faster than realtime on the GPU. Inference alone (encode plus decode) is 4.2x faster. The remaining 2.5 s is uploading the 3 GB model to VRAM, which is paid once per process, so a long-running server hides it.
- base.en is small enough that load and overhead dominate on both backends. The GPU still wins on the compute stages by about 3x.
- whisper.cpp ships no Windows Vulkan binary in any release, so this row needs CMake, the Visual Studio C++ build tools and the Vulkan SDK. The setup doc has the exact commands.

Raw data: [results/whispercpp/2026-10-04-vulkan-bench.csv](results/whispercpp/2026-10-04-vulkan-bench.csv) and [results/whispercpp/2026-10-04-cpu-bench.csv](results/whispercpp/2026-10-04-cpu-bench.csv).

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

1. SDXL on ComfyUI ROCm and on stable-diffusion.cpp, ROCm and Vulkan.
2. Fill the remaining Windows Vulkan rows: LM Studio, koboldcpp.
3. Retest Ollama and koboldcpp-rocm with `gfx1031` rocBLAS libraries. AMD's own `rocm-sdk-device-gfx1031` wheel may replace the community-patched bundles here too.
4. Linux dual boot with the `10.3.0` override: llama.cpp HIP, and PyTorch for a Linux-versus-Windows comparison now that both work.
5. Profile llama.cpp's Vulkan fallback matmul shaders on RDNA2 and report upstream.

## Contributing

Results from other RDNA2 cards are welcome. Open a pull request that adds your raw output under `results/<tool>/`, your headline rows to `results/summary.csv` with a `notes` entry naming the card, and a short hardware section to the setup doc if your path differed. Please include tool version and GPU driver version in every row.

## License

MIT. See [LICENSE](LICENSE).
