# ComfyUI on ROCm on Windows

Status: **works\***. ComfyUI runs on the RX 6700 XT with the ROCm PyTorch build from [pytorch-rocm-windows.md](pytorch-rocm-windows.md), no DirectML, no ZLUDA, no HIP SDK, no driver change. SD 1.5 at 512x512 samples at 7.0 steps/s, twice the best stable-diffusion.cpp result on the same card, and a 768x768 image takes about 10 s end to end. The asterisk: the torch install must be pinned by hand, and the first run at each new resolution or dtype takes minutes while MIOpen tunes its convolution kernels.

Tested: ComfyUI 0.38.0 (commit 5c460d8, 2026-10-04), torch 2.9.1+rocm7.13.0, torchvision 0.24.0+rocm7.13.0. Raw data: `results/comfyui/2026-10-05-comfyui-bench.csv`, server logs alongside it.

## Install

From the repo root. `src/` and `bin/` are git-ignored.

```powershell
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git src\ComfyUI
python -m venv bin\comfyui\.venv
$p = "bin\comfyui\.venv\Scripts\python.exe"

# 1. the ROCm torch stack, pinned, from AMD's index
& $p -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch==2.9.1+rocm7.13.0" "amd-torch-device-gfx1031==2.9.1+rocm7.13.0" "torchvision==0.24.0+rocm7.13.0" "amd-torchvision-device-gfx1031==0.24.0+rocm7.13.0"

# 2. ComfyUI's requirements from PyPI, with torch pinned so pip cannot swap in the CPU build
"torch==2.9.1+rocm7.13.0`ntorchvision==0.24.0+rocm7.13.0" | Set-Content bin\comfyui\constraints.txt
& $p -m pip install -c bin\comfyui\constraints.txt -r src\ComfyUI\requirements.txt
```

Step 2 matters. ComfyUI's requirements list `torch` and `torchvision` unversioned; without the constraints file pip resolves them against PyPI and replaces the ROCm build with the CPU one. ComfyUI does not need torchaudio, and there is no torchaudio 2.9.1 build in the index anyway.

Point ComfyUI at the checkpoint without copying it. `bin/comfyui/extra_model_paths.yaml`:

```yaml
rx6700xt:
  base_path: C:/Code/rx6700xt-local-ai/models
  checkpoints: .
```

Run:

```powershell
cd src\ComfyUI
..\..\bin\comfyui\.venv\Scripts\python.exe main.py --extra-model-paths-config ..\..\bin\comfyui\extra_model_paths.yaml --fp16-vae
```

The log should say `Device: cuda:0 AMD Radeon RX 6700 XT : native` and `Using sub quadratic optimization for attention`. Then open http://127.0.0.1:8188.

## The first run is slow, once

The first generation at a new resolution took 115 s at 512x512 and 250 s at 768x768, almost all of it in VAE decode (84 s and 183 s). That is MIOpen, the ROCm convolution library, searching for the best kernel per convolution shape. It writes what it finds to `~/.miopen/` and every later run at that shape is fast, including after a server restart. Switching the VAE to fp16 triggered a fresh search (80 s and 192 s) because the dtype is part of the shape.

Expect this once per resolution, per dtype, per model architecture. It is not a leak and not a bug in your setup. Do not benchmark the first run.

## Results

SD 1.5 fp16, 20 steps, cfg 7, seed 42, batch 1, 3 runs per config, first run discarded where it hit the MIOpen search. Steps/s from the sampler's own progress counter. "Total" is from prompt submission to image saved, with the server started with `--cache-none` so every run re-executes the whole graph; model weights stay in RAM.

| Config | Server flags | Steps/s | Sampling | Decode | Total |
|--------|--------------|--------:|---------:|-------:|------:|
| 512x512, euler_ancestral | default (sub-quadratic attention) | 7.0 | 3.3 s | 0.26 s | 4.4 s |
| 512x512, dpmpp_2m | default | 7.0 | 3.3 s | 0.26 s | 4.4 s |
| 512x512, euler_ancestral | `--use-pytorch-cross-attention` | 5.4 | 4.1 s | 0.26 s | 5.2 s |
| 512x512, euler_ancestral | `--use-split-cross-attention` | 6.3 | 3.6 s | 0.26 s | 4.6 s |
| 512x512, euler_ancestral | `--fp16-vae` | 7.0 | 3.2 s | 0.16 s | 4.2 s |
| 768x768, euler_ancestral | default | 2.27 | 9.2 s | 1.1 s | 11.2 s |
| 768x768, euler_ancestral | `--use-pytorch-cross-attention` | 1.1 | 18.6 s | 1.3 s | 20.8 s |
| 768x768, euler_ancestral | `--use-split-cross-attention` | 0.8 to 1.2 | 17 to 25 s | 1.5 s | 19 to 27 s |
| 768x768, euler_ancestral | `--fp16-vae` | 2.27 | 9.2 s | 0.33 s | 10.3 s |

Against the other image generators on this card, same model, same resolution:

| 20 steps | ComfyUI ROCm | sd.cpp ROCm | sd.cpp Vulkan |
|----------|-------------:|------------:|--------------:|
| 512x512 steps/s | 7.0 | 3.28 (with `--diffusion-fa`) | 2.08 |
| 768x768 steps/s | 2.27 | 1.18 (with `--diffusion-fa`) | 0.79 (with the buffer-size override) |

Takeaways:

- **Leave attention at the default.** ComfyUI picks its sub-quadratic attention on this card, and it beats both PyTorch's scaled-dot-product attention (23% slower at 512, 2x slower at 768) and split attention. The usual advice to force `--use-pytorch-cross-attention` on AMD does not apply here; this card has no flash-attention kernels for SDPA to dispatch to.
- **Use `--fp16-vae`.** Decode drops from 0.26 s to 0.16 s at 512 and from 1.1 s to 0.33 s at 768, sampling is unchanged, and the output is visually identical. ComfyUI defaults the VAE to fp32 on AMD because it cannot verify the fp16 path is NaN-free; it was fine here with the SD 1.5 VAE.
- **ComfyUI is 2x stable-diffusion.cpp.** The same weights on the same HIP runtime run twice as fast through PyTorch. The gap is the kernels: rocBLAS and MIOpen are tuned for this card, and ggml's own HIP kernels are not.
- The MIOpen warning `CK grouped conv library not found for device gfx1031` appears on the first sampler step. It means one kernel family is missing and MIOpen used another; nothing failed.

Samples: `results/comfyui/images/sample-*.png`.

## Running the bench

```powershell
bin\comfyui\.venv\Scripts\python.exe scripts\comfyui\bench.py --reps 3
bin\comfyui\.venv\Scripts\python.exe scripts\comfyui\bench.py --reps 3 "--label-suffix=-fp16vae" "--server-args=--fp16-vae"
```

The script starts the server itself, submits the workflow over the HTTP API, times each node over the websocket, parses steps/s from the server log, and stops the server. Pass `--label-suffix` and `--server-args` with `=` because their values start with a dash.

## Not tested yet

- `torch.compile`. The index has a `triton` package; ComfyUI reports the Triton backend missing. Untested.
- SDXL and anything larger than SD 1.5.
- LoRA, ControlNet, upscalers: all plain PyTorch, so expected to work, but no numbers.
