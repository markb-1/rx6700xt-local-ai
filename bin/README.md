# bin/

Drop tool binaries here. This folder is git-ignored apart from this file.

Expected layout (one subfolder per tool and build):

```
bin/
  llama.cpp/
    b11146-vulkan/      llama-bench.exe, llama-server.exe, ggml-vulkan.dll, ...
```

The harness scripts take a `-Bin` parameter, so any layout works. This one just keeps versions side by side.

The stable-diffusion.cpp ROCm build needs a HIP runtime that lives here too, as a Python venv of AMD's ROCm wheels:

```
bin/
  stable-diffusion.cpp/
    master-929-vulkan/  sd-cli.exe, ggml-vulkan.dll, ...
    master-929-rocm/    sd-cli.exe, stable-diffusion.dll (needs the venv below on PATH)
  therock-7.14.0/
    .venv/              pip install of rocm[libraries]==7.14.0 and rocm-sdk-device-gfx1031, about 2.8 GB
  therock-torch/
    .venv/              torch 2.9.1+rocm7.13.0 and torchvision with their gfx1031 device packages, about 3.5 GB
```

See `docs/setup/stable-diffusion-cpp-rocm.md` and `scripts/sdcpp/rocm-env.ps1`.
