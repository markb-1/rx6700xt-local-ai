# bin/

Drop tool binaries here. This folder is git-ignored apart from this file.

Expected layout (one subfolder per tool and build):

```
bin/
  llama.cpp/
    b11146-vulkan/      llama-bench.exe, llama-server.exe, ggml-vulkan.dll, ...
```

The harness scripts take a `-Bin` parameter, so any layout works. This one just keeps versions side by side.
