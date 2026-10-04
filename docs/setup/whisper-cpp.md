# whisper.cpp

Status on Windows: **CPU works**, **Vulkan needs a source build**.

whisper.cpp publishes Windows binaries for CPU, OpenBLAS and CUDA only. There is no Vulkan zip in any release, so the GPU path on this card means building from source with the Vulkan SDK. That build is on the roadmap. The CPU numbers below are the baseline it will be compared against.

## Install (CPU build)

1. Download `whisper-bin-x64.zip` from the whisper.cpp releases page. Note that the release tags with binaries are the `bNNNN` build tags, not the `vX.Y.Z` version tags.
2. Unzip into `bin/whisper.cpp/<build>-cpu/`. Binaries are under `Release/`.
3. Use `whisper-cli.exe` and `whisper-bench.exe`. The old `main.exe` and `bench.exe` are stubs that only print a deprecation warning and exit 1.
4. Models come from the `ggerganov/whisper.cpp` repo on Hugging Face: `ggml-base.en.bin` (148 MB) and `ggml-large-v3.bin` (3.1 GB).
5. The test clip is `samples/jfk.wav` from the whisper.cpp repo, 11 seconds of speech.

```powershell
.\bin\whisper.cpp\b5130-cpu\Release\whisper-cli.exe -m .\models\ggml-base.en.bin -f .\models\samples\jfk.wav -t 8
```

## Building the Vulkan backend

Needs CMake, the Visual Studio C++ build tools and the LunarG Vulkan SDK (for `glslc`). Then:

```powershell
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

Not yet done on the test machine. Results will be added as a second row when it is.

## Running the bench

```powershell
cd scripts\whispercpp
.\bench.ps1 -Summary ..\..\results\summary.csv -GpuDriver 32.0.21045.5002
```

Runs each model three times on the JFK clip and records load, encode, decode and total time. Pass `-Bin` and `-Backend vulkan` to benchmark a GPU build with the same script.
