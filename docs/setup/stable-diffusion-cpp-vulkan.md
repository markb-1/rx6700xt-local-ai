# stable-diffusion.cpp with the Vulkan backend

Status: **works** at 512x512 with no configuration. **works\*** at 768x768 and above: the Windows AMD Vulkan driver caps single buffers at 2 GiB, and any tensor larger than that is silently run on the CPU. See [the 2 GiB buffer limit](#the-2-gib-buffer-limit-768x768-and-above) for the three fixes.

## Install

1. Download `sd-master-<commit>-bin-win-vulkan-x64.zip` from the stable-diffusion.cpp releases page.
2. Unzip into `bin/stable-diffusion.cpp/<release>-vulkan/`.
3. Put a checkpoint in `models/`. For the baseline test this repo uses Stable Diffusion 1.5 as `v1-5-pruned-emaonly-fp16.safetensors` (2.1 GB) from the `Comfy-Org/stable-diffusion-v1-5-archive` repo on Hugging Face.
4. Smoke test:

```powershell
.\bin\stable-diffusion.cpp\master-929-vulkan\sd-cli.exe -m .\models\v1-5-pruned-emaonly-fp16.safetensors -p "a lighthouse on a rocky coast at sunset" -o test.png -v
```

The log should show `ggml_vulkan: 0 = AMD Radeon RX 6700 XT` with `matrix cores: none`. That last part is expected on RDNA2.

## What the auto-fit does

Build master-929 picks device placement itself. With SD 1.5 it puts the text encoder, UNet and VAE all on the GPU, about 2 GB of weights plus a 560 MB UNet compute buffer and a 2 GB VAE decode buffer at 512x512. Total is well inside 12 GB. Larger models (SDXL, FLUX) will need `--offload-to-cpu` or `--backend vae=cpu`; those rows are still to be tested.

The auto-fit also cuts a graph between devices per operation. If one tensor cannot be allocated on the GPU at all, the operations that touch it run on the CPU and the log reports two compute buffers for that stage, one `(VRAM) on Vulkan0` and one `(RAM) on CPU`. The only other hint is a single warning line, `ggml_vulkan: Failed to allocate pinned memory (Requested buffer size exceeds device buffer size limit)`. Generation still succeeds, just slowly. The bench script records both buffers so a CPU fallback shows up in the CSV.

## Flags that matter on this card

| Flag | Notes |
|------|-------|
| `--diffusion-fa` | Flash attention in the UNet. 40% slower at 512x512, but at 768x768 and above it is what keeps attention on the GPU. See below |
| `--vae-conv-direct` | Direct convolutions in the VAE instead of im2col. Needed at 768x768 and above, and faster there (2.1 s against 6.0 s decode) |
| `--vae-tiling` | Tiled VAE decode. Also avoids the limit, 3.5 s decode at 768, but `--vae-conv-direct` is faster and needs no tile seams |
| `GGML_VK_FORCE_MAX_BUFFER_SIZE` | Environment variable, not a flag. Overrides the 2 GiB cap; see below |
| `-t 8` | CPU threads for model loading only; compute is on the GPU, unless something fell back to CPU |
| `--backend vae=cpu` | Moves VAE decode to CPU if VRAM is tight at high resolution |

## Running the bench

```powershell
cd scripts\sdcpp
.\bench.ps1 -Summary ..\..\results\summary.csv -GpuDriver 32.0.21045.5002
```

Default configs: 512x512 euler_a 20 steps, the same with flash attention, 512x512 dpm++2m, 768x768 euler_a as shipped (hits the CPU fallback), and 768x768 euler_a with `--diffusion-fa --vae-conv-direct`. Three reps each, fixed seed 42. Images land in `results/sdcpp/images/` (git-ignored apart from the samples). The CSV has `unet_cpu_buffer_mb` and `vae_cpu_buffer_mb` columns; any value there means that stage ran partly on the CPU.

To bench with the environment-variable fix, set it in the shell first:

```powershell
$env:GGML_VK_FORCE_MAX_BUFFER_SIZE = 3221225472
$env:GGML_VK_FORCE_MAX_ALLOCATION_SIZE = 3221225472
.\bench.ps1 -Configs @(@{ label = '768-euler_a-20-forcemax3g'; width = 768; height = 768; steps = 20; sampler = 'euler_a'; args = @() })
```

## The 2 GiB buffer limit (768x768 and above)

**Symptom.** SD 1.5 at 768x768 takes 290 s for 20 sampling steps against 10 s at 512x512, and VAE decode takes 6.5 s against 0.9 s. Reported buffers stay under 6 GB, so it is not VRAM exhaustion.

**Cause.** The AMD Windows driver reports `maxMemoryAllocationSize` and `maxBufferSize` of `0x80000000`, exactly 2 GiB, for this card (`vulkaninfo`). Two tensors cross that line at 768x768:

- The UNet's self-attention score matrix at the top resolution is 9216 tokens x 9216 tokens x 8 heads in fp32, which is 2592 MB. At 512x512 it is 4096 x 4096 x 8, which is 512 MB and matches the 560 MB UNet buffer.
- The VAE decoder's last 3x3 convolutions use im2col, which expands a 768 x 768 x 128 activation by 9x to about 2.7 GB. At 512x512 that is 1.2 GB and fits.

ggml refuses to allocate a single tensor larger than the device's maximum buffer size, so the auto-fit places those operations on the CPU. The verbose log shows it directly:

```
ggml_vulkan: Failed to allocate pinned memory (Requested buffer size exceeds device buffer size limit: ErrorOutOfDeviceMemory)
unet compute buffer size: 283.96 MB(VRAM) on Vulkan0 (peak across 1 segment)
unet compute buffer size: 2626.12 MB(RAM) on CPU (peak across 1 segment)
```

At 512x512 the UNet has a single 560 MB buffer in VRAM and no CPU buffer. A 640x640 run (6400 tokens, 1310 MB score matrix) also stays on the GPU, so this is a cliff at the 2 GiB line, not a curve.

Upstream context: llama.cpp issue [#15054](https://github.com/ggml-org/llama.cpp/issues/15054) reports the same 2 GiB `maxMemoryAllocationSize` on AMD's own Vulkan driver (AMDVLK on Linux; the Mesa RADV driver does not have the cap). PR [#15815](https://github.com/ggml-org/llama.cpp/pull/15815) lets ggml split a graph across several buffers, which is why the rest of the UNet still fits, but one tensor still cannot exceed the limit.

**Fixes, measured at 768x768, euler_a, 20 steps, 3 runs each:**

| Config | Sampling | VAE decode | Wall | Where it runs |
|--------|---------:|-----------:|-----:|---------------|
| as shipped | 289.8 s | 6.5 s | 297 s | UNet attention and VAE conv on CPU |
| `--diffusion-fa --vae-conv-direct` | 43.0 s | 1.3 s | 44.9 s | all GPU |
| `GGML_VK_FORCE_MAX_BUFFER_SIZE=3221225472` (and `..._MAX_ALLOCATION_SIZE`) | 25.3 s | 1.7 s | 27.7 s | all GPU, 5.3 GB UNet buffer |

1. **Flags only: `--diffusion-fa --vae-conv-direct`.** Flash attention never materialises the score matrix, and direct convolution never builds the im2col tensor, so nothing exceeds 2 GiB. 6.6x faster than shipped. Flash attention is still the slow attention path on RDNA2, which is why this is not the fastest option. This also works at 1024x1024, where the score matrix would be 8.6 GB and no override could help: 5.6 s per step, 2.2 s decode, all on the GPU.
2. **Environment variable: `GGML_VK_FORCE_MAX_BUFFER_SIZE=3221225472` plus `GGML_VK_FORCE_MAX_ALLOCATION_SIZE=3221225472`.** Tells ggml to ignore the reported limit. The driver's 2 GiB figure is advisory and the allocation succeeds: the UNet runs in one 5.3 GB VRAM buffer with the default attention. 10.7x faster than shipped and the fastest 768 path. Unchanged at 512x512 (2.14 steps/s against 2.08). Strictly this allocates beyond what the driver promises, so if a future driver starts honouring the cap, expect an allocation failure rather than a slowdown. Set it to 3 GiB rather than higher; the 768 tensors are 2.6 to 2.7 GB and the VAE buffer under this setting is already 5.7 GB.
3. **`--vae-tiling`** fixes only the VAE half (3.5 s decode). Use it if `--vae-conv-direct` ever misbehaves.

All three paths produce the same image as the CPU fallback for the same seed, up to the small numeric differences between CPU and GPU kernels. Sample: `results/sdcpp/images/sample-sd15-768-euler_a-20-forcemax3g.png`. Raw data: `results/sdcpp/2026-10-05-sd15-vulkan-768-fallback.csv`, and the shipped-config log at `results/sdcpp/2026-10-05-sd15-vulkan-768-2steps-cpu-fallback.log`.

## Known issues

- **`--diffusion-fa` is 40% slower** than the default attention at 512x512 on this card, though it uses far less VRAM. See the README results. Still unexplained; the leading guess is that the Vulkan flash-attention shader is tuned for cards with matrix cores.
- **768x768 and above silently fall back to CPU** for the largest tensors. Fixed by the flags or the environment variable above. The silence is the real problem: one warning line, then a 30x slowdown reported as success.

## The ROCm build

The same release ships `sd-master-<commit>-bin-win-rocm-7.14.0-x64.zip`. It fails out of the box with `0xC0000135` because it needs the HIP 7 runtime, but that runtime is a pip install away and the build is then faster than Vulkan on this card, including at 768x768 where HIP has no 2 GiB buffer cap. See [stable-diffusion-cpp-rocm.md](stable-diffusion-cpp-rocm.md).
