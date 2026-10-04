# stable-diffusion.cpp with the Vulkan backend

Status: **works**. Prebuilt Windows Vulkan binaries run on the RX 6700 XT with no configuration.

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

## Flags that matter on this card

| Flag | Notes |
|------|-------|
| `--diffusion-fa` | Flash attention in the UNet. Measured in the bench; see results |
| `--vae-conv-direct` | Direct convolutions in the VAE. Untested here |
| `-t 8` | CPU threads for model loading only; compute is on the GPU |
| `--backend vae=cpu` | Moves VAE decode to CPU if VRAM is tight at high resolution |

## Running the bench

```powershell
cd scripts\sdcpp
.\bench.ps1 -Summary ..\..\results\summary.csv -GpuDriver 32.0.21045.5002
```

Default configs: 512x512 euler_a 20 steps, the same with flash attention, 512x512 dpm++2m, and 768x768 euler_a. Three reps each, fixed seed 42. Images land in `results/sdcpp/images/` (git-ignored apart from one sample).

## Known issues

- **`--diffusion-fa` is 40% slower** than the default attention at 512x512 on this card, though it uses far less VRAM. See the README results.
- **768x768 is about 30x slower than 512x512**, 290 s against 10 s for the sampling stage, and VAE decode slows 7x. Reported buffers total under 5 GB so it is not VRAM exhaustion. Unexplained so far. Generate at 512 and upscale until this is understood.

## The ROCm build

The same release ships `sd-master-<commit>-bin-win-rocm-7.14.0-x64.zip`. On this machine it fails to start with Windows error `0xC0000135` (STATUS_DLL_NOT_FOUND). The 1 GB `stable-diffusion.dll` imports `amdhip64_7.dll`, the HIP 7 runtime, and the Adrenalin driver only installs `amdhip64.dll` and `amdhip64_6.dll`.

Two things make this worth a retest:

- Scanning the DLL shows it bundles kernels for `gfx1031` alongside `gfx1030`, `gfx1032` and the RDNA3 and RDNA4 targets. So unlike most ROCm builds, this one was compiled for the 6700 XT.
- The missing piece is only the HIP 7 runtime, which ships with the AMD HIP SDK for Windows.

Installing the HIP SDK 7.x and retrying is on the roadmap.
