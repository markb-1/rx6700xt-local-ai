# PyTorch on ROCm on Windows

Status: **works\***. PyTorch 2.9.1 with HIP sees the RX 6700 XT, matmuls match the CPU, autograd and training work, and fp16 GEMM runs at 21 TFLOPS, about 80% of the card's theoretical peak. The asterisk is because the install comes from AMD's ROCm wheel index rather than pytorch.org, and because the version pairing between torch and torchvision has to be done by hand.

AMD's HIP SDK docs still list this card as unsupported on Windows, and pytorch.org offers no ROCm build for Windows at all. Neither matters: AMD's TheRock build system publishes Windows wheels of torch for every RDNA2 target, kernels included.

## Install

Python 3.12 (3.10 to 3.14 have wheels). From the repo root:

```powershell
python -m venv bin\therock-torch\.venv
bin\therock-torch\.venv\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch==2.9.1+rocm7.13.0" "amd-torch-device-gfx1031==2.9.1+rocm7.13.0" "torchvision==0.24.0+rocm7.13.0" "amd-torchvision-device-gfx1031==0.24.0+rocm7.13.0"
```

That pulls in `rocm-sdk-core`, `rocm-sdk-libraries` and `rocm-sdk-device-gfx1031` at 7.13.0 as dependencies, about 1 GB of downloads and 3.5 GB on disk. Nothing touches the system PATH; `rocm-bootstrap` wires the DLLs up at import time. Replace `gfx1031` with your target; the index has device packages for `gfx1030` through `gfx1036`.

Check:

```powershell
bin\therock-torch\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.9.1+rocm7.13.0 7.13.99004-3309c611 True AMD Radeon RX 6700 XT`. The CUDA API names are normal; HIP is exposed through `torch.cuda`.

## Pin the versions

Two traps, both hit while setting this up:

1. **torchvision must match torch's release, and its ROCm suffix.** A bare `pip install torchvision` from this index chose 0.27.0+rocm7.14.1 next to torch 2.9.1+rocm7.13.0. It imports and then fails on the first model with `RuntimeError: operator torchvision::nms does not exist`, because its C++ extension silently failed to load. torch 2.9 pairs with torchvision 0.24. Pin both the version and the `+rocm` suffix.
2. **Each package has a device twin.** `torch` is the host code; `amd-torch-device-gfx1031` holds the compiled kernels for this card, and `amd-torchvision-device-gfx1031` likewise for torchvision's ops. TheRock's documented shorthand `torch[device-gfx1031]` expands to the same thing. Without the device package you get `torch.cuda.is_available()` true and kernel-not-found errors later.

The index also has torch 2.9.1 only at rocm7.13.0. Newer ROCm suffixes (7.14.0, 7.14.1) carry newer torch releases. Pick one ROCm suffix and stay on it for every package.

## Results

`scripts/pytorch/bench.py`, 10 reps per test after warmup, RX 6700 XT, driver 32.0.21045.5002. Raw CSV: `results/pytorch/2026-10-05-torch-bench.csv`.

| Test | Result | Notes |
|------|-------:|-------|
| `torch.cuda.is_available()` | True | gfx1031, 12272 MiB, 20 CUs |
| 512x512 fp32 matmul vs CPU | max diff 1e-4 | correct |
| autograd | ok | d/dx sum(x²) = 2x |
| 4096x4096 GEMM fp16 | 21.1 TFLOPS | 6.5 ms. Card peak is 26.4 |
| 4096x4096 GEMM fp32 | 11.5 TFLOPS | 11.9 ms. Card peak is 13.2 |
| 4096x4096 GEMM bf16 | 6.0 TFLOPS | RDNA2 has no bf16 hardware; emulated. Use fp16 |
| conv 3x3, 128ch, 256px, fp16 | 0.66 ms | MIOpen |
| MLP train step, batch 256, Adam, fp32 | 2.9 ms | |
| ResNet-18 forward, fp16, batch 8 | 2.6 ms | torchvision 0.24.0. First few passes are 2 to 3x slower while MIOpen picks kernels |
| ResNet-18 train step, fp32, batch 8, SGD | 21.0 ms | |
| torchvision NMS custom op | ok | runs on device |

Takeaways:

- GEMM efficiency is 80 to 87% of theoretical, so rocBLAS has real tuned kernels for gfx1031 here, not fallbacks.
- bf16 is half the speed of fp32. Mixed precision on this card means fp16, with loss scaling.
- A warning on import, `MIOpen(HIP): Warning [OpenRuntimeLibraryForDevice] CK grouped conv library not found for device gfx1031`, means the composable-kernel conv path is absent for this target and MIOpen uses its other solvers. Convolutions still work; grouped or depthwise convs may be slower than on supported cards. Two `xnack 'Off' was requested` warnings are harmless.

## What this opens up

- **Training on this card on Windows.** Previously thought to need Linux with the gfx1030 override.
- **ComfyUI on ROCm** instead of DirectML or ZLUDA. Next row to test. The MIOpen warning above is the thing to watch for in UNet-heavy workloads.
- **Anything else on PyTorch**: whisper (the original), diffusers, transformers. All untested.

## Running the bench

```powershell
bin\therock-torch\.venv\Scripts\python.exe scripts\pytorch\bench.py
```

Writes `results/pytorch/<date>-torch-bench.csv` with one row per test. Failures are recorded as rows with `status=fail` and the error, so a partial run still produces a record.
