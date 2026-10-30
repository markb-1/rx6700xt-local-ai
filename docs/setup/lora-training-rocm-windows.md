# LoRA fine-tuning on ROCm on Windows

Status: **works\***. Hugging Face `transformers` and `peft` train LoRA adapters on the RX 6700 XT on Windows, on the same TheRock PyTorch build as the [PyTorch row](pytorch-rocm-windows.md). Tested with Qwen2.5-Coder-1.5B-Instruct and Llama 3.2 3B. The asterisk covers three things: training has to be fp16 rather than bf16, a 3B model needs gradient checkpointing to fit in 12 GB, and the extra packages have to be installed without letting pip replace the ROCm torch.

No throughput numbers yet. The runs so far were not made with a script in this repo, so speed is left out until `scripts/pytorch/` has a LoRA benchmark (see the roadmap).

## Install

Start from the PyTorch venv in [pytorch-rocm-windows.md](pytorch-rocm-windows.md), then add the Hugging Face stack from PyPI with torch pinned, so pip cannot swap in a CPU build:

```powershell
"torch==2.9.1+rocm7.13.0" | Out-File -Encoding ascii bin\therock-torch\constraints.txt
bin\therock-torch\.venv\Scripts\python.exe -m pip install -c bin\therock-torch\constraints.txt "transformers==4.57.6" "peft==0.21.2" "accelerate==1.15.0" safetensors
```

Versions tested: transformers 4.57.6, peft 0.21.2, accelerate 1.15.0, torch 2.9.1+rocm7.13.0.

## What fits in 12 GB

LoRA rank 16 on all seven linear projections of every layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), base model loaded in fp16, AdamW, effective batch 16.

| Model | Max sequence | Batch x accumulation | Gradient checkpointing | Result |
|-------|-------------:|---------------------:|:----------------------:|--------|
| Qwen2.5-Coder-1.5B-Instruct | 384 | 8 x 2 | off | fits |
| Qwen2.5-Coder-1.5B-Instruct | 384 | 16 x 1 | off | out of memory |
| Llama 3.2 3B | 640 | 8 x 2 | off | out of memory (tried to allocate 22 GB) |
| Llama 3.2 3B | 640 | 4 x 4 | on | fits, about 11.4 GB in use |

Gradient checkpointing with a frozen base needs `model.enable_input_require_grads()` after `model.gradient_checkpointing_enable()`, before wrapping the model with `get_peft_model`. Without it the checkpointed activations do not require grad and backward fails.

## Precision

RDNA2 has no bf16 hardware (bf16 GEMM runs at half the fp32 rate, see the PyTorch row), so use fp16:

- Load the base with `torch_dtype=torch.float16`.
- Run the forward pass under `torch.autocast("cuda", dtype=torch.float16)` and use `torch.amp.GradScaler("cuda")`.
- PEFT keeps the adapter weights in fp32 by default, which is what lets the gradient scaler unscale them. Do not cast the whole PEFT model to fp16.

## Known issues

- **Llama 3 has no pad token.** Set `tok.pad_token = tok.eos_token` and mask padded positions out of the loss.
- **Use the model's own end token in the training targets.** Qwen chat models end a turn with `<|im_end|>`, Llama 3 base models end text with `<|end_of_text|>`. A hard-coded end string from another family trains the model to type it as text and never stop.
- **Merge on the CPU in fp32.** `PeftModel.from_pretrained(base_fp32, adapter).merge_and_unload()` needs about 4 bytes per parameter of system RAM and disk (13 GB for a 3B model); merging into the fp16 base rounds the update twice.

## Running the result with llama.cpp

The merged model converts and runs on the [llama.cpp Vulkan](llama-cpp-vulkan.md) build like any other:

```powershell
python <llama.cpp source>\convert_hf_to_gguf.py <merged_dir> --outtype f16 --outfile model-f16.gguf
bin\llama.cpp\b11146-vulkan\llama-quantize.exe model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

Alternatively keep the adapter separate: `convert_lora_to_gguf.py` turns it into a GGUF adapter, and `llama-server --lora adapter.gguf` applies it at run time on Vulkan. Both paths work on this card at b11146.
