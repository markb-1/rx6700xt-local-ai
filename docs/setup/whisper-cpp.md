# whisper.cpp

Status on Windows: **CPU works** from the prebuilt zip, **Vulkan works** from a source build.

whisper.cpp publishes Windows binaries for CPU, OpenBLAS and CUDA only. There is no Vulkan zip in any release, so the GPU path on this card means building from source with the Vulkan SDK. The build is straightforward once the toolchain is installed (steps below) and on this card it runs large-v3 inference about 4x faster than the CPU build. The first run in a process spends roughly 2.5 s compiling Vulkan shader pipelines; later runs do not.

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

Needs CMake, the Visual Studio 2022 C++ build tools and the LunarG Vulkan SDK (for `glslc`). All three install cleanly with winget or the Microsoft bootstrapper:

```powershell
winget install --id Kitware.CMake -e
winget install --id KhronosGroup.VulkanSDK -e
# Build Tools: download https://aka.ms/vs/17/release/vs_BuildTools.exe then
.\vs_BuildTools.exe --quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
```

Then, from a shell where `VULKAN_SDK` points at the SDK folder:

```powershell
git clone --depth 1 --branch b5130 https://github.com/ggml-org/whisper.cpp src\whisper.cpp
cd src\whisper.cpp
cmake -B build-vulkan -G "Visual Studio 17 2022" -A x64 -DGGML_VULKAN=ON -DWHISPER_SDL2=OFF
cmake --build build-vulkan --config Release --target whisper-cli whisper-bench
```

Binaries land in `build-vulkan\bin\Release\`. Copy that folder to `bin\whisper.cpp\b5130-vulkan\` so the bench script's `-Bin` default layout matches. The `src/` folder is git-ignored.

Versions used on the test machine: CMake 4.4.3, Vulkan SDK 1.4.363.0, Visual Studio 2022 Build Tools (MSVC v143).

## Running the bench

```powershell
cd scripts\whispercpp
.\bench.ps1 -Summary ..\..\results\summary.csv -GpuDriver 32.0.21045.5002
```

Runs each model three times on the JFK clip and records load, encode, decode and total time. Pass `-Bin` and `-Backend vulkan` to benchmark a GPU build with the same script.
