# stable-diffusion.cpp with the ROCm (HIP) backend on Windows

Status: **works\***. The prebuilt Windows ROCm binary runs on the RX 6700 XT and is faster than the Vulkan build, but it needs a HIP runtime that no installer provides for this card. The runtime comes from AMD's ROCm Python wheels instead, with no display driver change.

Tested: release master-929 (commit 3f8527a), asset `sd-master-3f8527a-bin-win-rocm-7.14.0-x64.zip`, with ROCm 7.14.0 wheels. Results are in the README and `results/sdcpp/2026-10-05-sd15-rocm-bench.csv`.

## Why the zip does not run on its own

The zip holds only `sd-cli.exe`, `sd-server.exe` and a 1 GB `stable-diffusion.dll`. That DLL imports `amdhip64_7.dll` (the HIP 7 runtime) and `hipblas.dll`, and neither ships with it. The Adrenalin driver installs `amdhip64.dll` and `amdhip64_6.dll` only, so the exe dies at startup with `0xC0000135` (STATUS_DLL_NOT_FOUND).

The obvious fix, AMD's HIP SDK for Windows, is the wrong one here:

- HIP SDK 7.1.1 and 7.2 list every RX 6000 card as unsupported, with no runtime support, so their rocBLAS has no `gfx1031` kernels.
- The SDK installer bundles an AMD Software PRO display driver, and the silent install cannot deselect it. That would change the driver every other row in this repo was measured with.
- The `7.14.0` in the asset name is not a HIP SDK version. It is a TheRock build.

## Where the binary's runtime actually comes from

The stable-diffusion.cpp CI builds the Windows ROCm binary against AMD's ROCm Python wheels from TheRock (`.github/workflows/build.yml`, job `windows-latest-rocm`): it runs `pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,devel]==7.14.0"` and compiles with `GPU_TARGETS` that include `gfx1030` through `gfx1036`, so the DLL carries kernels for this card. The same index publishes the matching runtime, and a per-GPU package with the rocBLAS kernels. Installing those two gives the exact runtime the binary was linked against.

## Install

1. Download `sd-master-<commit>-bin-win-rocm-<version>-x64.zip` from the releases page and unzip it into `bin/stable-diffusion.cpp/<release>-rocm/`. Note the `<version>`; the wheels must match it.
2. Install Python if the machine has none. This one got 3.12 from winget, user scope, no admin:

```powershell
winget install --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
```

3. Create a venv under `bin/` (git-ignored) and install the runtime and the `gfx1031` kernel package. About 900 MB of downloads, 2.8 GB on disk:

```powershell
python -m venv bin\therock-7.14.0\.venv
bin\therock-7.14.0\.venv\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries]==7.14.0" "rocm-sdk-device-gfx1031==7.14.0"
```

   Replace `gfx1031` with your card's target. The index has `rocm-sdk-device-gfx1030` through `gfx1036` for the rest of RDNA2.

4. Put the two wheel `bin` folders on PATH for the session, then run as usual. `scripts/sdcpp/rocm-env.ps1` does the PATH part:

```powershell
. .\scripts\sdcpp\rocm-env.ps1
.\bin\stable-diffusion.cpp\master-929-rocm\sd-cli.exe -m .\models\v1-5-pruned-emaonly-fp16.safetensors -p "a lighthouse on a rocky coast at sunset" -o test.png -v
```

The log should start with `ggml_cuda_init: found 1 ROCm devices` and `Device 0: AMD Radeon RX 6700 XT, gfx1031`. The `ggml_cuda` prefix is normal; the HIP backend is the CUDA backend compiled for AMD.

## What goes wrong without the device package

With `rocm[libraries]` alone the device initialises and the model loads, then the first matrix multiply aborts:

```
rocBLAS error: Cannot read ...\_rocm_sdk_libraries\bin\/rocblas/library/TensileLibrary.dat: No such file or directory for GPU arch : gfx1031
```

`rocm-sdk-device-gfx1031` drops 142 files (23 MB) into `_rocm_sdk_libraries/bin/rocblas/library/`, including `TensileLibrary_lazy_gfx1031.dat`, and rocBLAS finds them without any environment variable. This is the same failure the community "patched rocBLAS" bundles exist to fix for Ollama and koboldcpp-rocm; here AMD's own wheels supply the kernels.

## Flags that matter on this card

| Flag | ROCm | Vulkan, for contrast |
|------|------|----------------------|
| `--diffusion-fa` | **Use it.** 24% faster at 512x512 (3.28 against 2.64 steps/s) and 54% faster at 768x768 (1.18 against 0.77). The HIP flash-attention kernels suit this card | 40% slower at 512x512 |
| `--vae-conv-direct` | **Avoid.** VAE decode goes from 1.6 s to 17.9 s at 768x768 | 2 to 3x faster decode at 768 and required there |
| `--vae-tiling` | Untested on ROCm | Works |
| 768x768 as shipped | Fine: 0.77 steps/s with a 2.7 GB UNet buffer in VRAM. HIP has no 2 GiB per-buffer cap | 0.07 steps/s unless the 2 GiB cap is worked around |
| `-t 8` | Loading only | Loading only |

Text encoding is slower on ROCm, 0.73 s against 0.28 s per image. It is a fixed cost and small next to sampling.

## Running the bench

```powershell
. .\scripts\sdcpp\rocm-env.ps1
cd scripts\sdcpp
.\bench.ps1 -Bin ..\..\bin\stable-diffusion.cpp\master-929-rocm -Images ..\..\results\sdcpp\images\rocm -Summary ..\..\results\summary.csv -GpuDriver 32.0.21045.5002
```

The script labels the backend `rocm` from the `found N ROCm devices` log line.

## Known issues

- **Version coupling.** The wheels must be the version in the asset name. The index also has 7.14.1 and 7.13.0; mixing is untested. Each new stable-diffusion.cpp release may move to a new TheRock version and need a fresh venv.
- **`--vae-conv-direct` is very slow** on the HIP backend here, so the Vulkan advice for 768x768 does not carry over. Use `--diffusion-fa` alone.
- **Not a system install.** Nothing goes on the system PATH, so every shell that runs the ROCm build needs the env script first. That is deliberate: the same `amdhip64_7.dll` name could collide with a future driver or SDK.
- **Output matches Vulkan.** For the same seed the ROCm and Vulkan builds produce the same composition at both 512x512 and 768x768, with only kernel-level numeric differences. Compare `results/sdcpp/images/sample-sd15-512-euler_a-20.png` with the `-rocm` sample.

## Implications for the rest of the matrix

The same wheel index carries `amd-torch-device-gfx1031` and `amd-torchvision-device-gfx1031` for Windows, which suggests PyTorch on ROCm might also work on this card without Linux. That is the next thing to test. See the README roadmap.
