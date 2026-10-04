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

## Why this card is awkward

AMD's ROCm docs list the RX 6700 XT as **runtime only** on Windows: the HIP and OpenCL runtimes load, but there is no HIP SDK support, so nothing that ships ROCm kernels (PyTorch, rocBLAS-based inference) works out of the box. On Linux the card is also unsupported officially but runs under `HSA_OVERRIDE_GFX_VERSION=10.3.0`, borrowing the `gfx1030` (RX 6800/6900) kernels.

Consequences:

- Vulkan is the default working backend on Windows for llama.cpp, whisper.cpp and stable-diffusion.cpp.
- RDNA2 has no cooperative-matrix (tensor-core style) Vulkan extension, so llama.cpp uses its fallback matmul shaders. Expect lower prompt-processing speed than RDNA3 at the same memory bandwidth.
- ROCm-based Windows tools (Ollama ROCm, koboldcpp-rocm, ZLUDA) need community-rebuilt rocBLAS libraries for `gfx1031`.
- Stock Ollama 0.6.7 on this machine fell back to CPU, then failed with `RMS_NORM failed` when forced onto ROCm. Retest pending with patched libraries.
