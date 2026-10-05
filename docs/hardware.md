# Test rig

All results in this repo come from this machine unless a result file says otherwise.

| Component | Detail |
|-----------|--------|
| GPU | AMD Radeon RX 6700 XT, 12 GB GDDR6, RDNA2, LLVM target `gfx1031` |
| GPU driver | AMD Adrenalin, Windows driver 32.0.21045.5002 (August 2026) |
| Vulkan | API 1.4.315, driver 2.0.353 |
| CPU | AMD Ryzen 7 5800X, 8 cores / 16 threads |
| RAM | 32 GB DDR4 |
| OS | Windows 11 Home 24H2 (build 26200) |
| Build toolchain | CMake 4.4.3, Vulkan SDK 1.4.363.0, Visual Studio 2022 Build Tools (MSVC v143) |

## Why this card is awkward

AMD's HIP SDK for Windows lists the RX 6700 XT as **unsupported** from 7.1.1 onward (earlier releases said runtime only): the SDK's rocBLAS ships no `gfx1031` kernels, so nothing built on the SDK (PyTorch, rocBLAS-based inference) works out of the box. On Linux the card is also unsupported officially but runs under `HSA_OVERRIDE_GFX_VERSION=10.3.0`, borrowing the `gfx1030` (RX 6800/6900) kernels.

There is a loophole. AMD's newer ROCm build system, TheRock, publishes Python wheels at `https://repo.amd.com/rocm/whl-multi-arch/` with a per-GPU kernel package for every RDNA2 target, `rocm-sdk-device-gfx1031` included, plus PyTorch wheels per target. Tools built against those wheels, like the stable-diffusion.cpp Windows ROCm binary, run on this card once the matching wheels are installed. See [docs/setup/stable-diffusion-cpp-rocm.md](setup/stable-diffusion-cpp-rocm.md).

Consequences:

- Vulkan is the default working backend on Windows for llama.cpp, whisper.cpp and stable-diffusion.cpp.
- RDNA2 has no cooperative-matrix (tensor-core style) Vulkan extension, so llama.cpp uses its fallback matmul shaders. Expect lower prompt-processing speed than RDNA3 at the same memory bandwidth.
- ROCm-based Windows tools built against the HIP SDK (Ollama ROCm, koboldcpp-rocm, ZLUDA) need rocBLAS libraries for `gfx1031` from somewhere else: community rebuilds, or possibly AMD's own wheel.
- Stock Ollama 0.6.7 on this machine fell back to CPU, then failed with `RMS_NORM failed` when forced onto ROCm. Retest pending with patched libraries.
