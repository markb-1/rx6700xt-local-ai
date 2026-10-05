"""Probe and micro-benchmark PyTorch on the GPU, writing one CSV row per test.

Usage (from the repo root, inside a venv that has torch):
    python scripts/pytorch/bench.py [--out results/pytorch/<file>.csv] [--reps 10]

Checks, in order: torch sees a device, a matmul on the device matches the CPU,
backward works, then timed fp16/fp32 GEMM (TFLOPS), a 3x3 conv (ms) and a small
training step (ms). Any failure is recorded as a row with status=fail and the
error text, and the script carries on to the next test.
"""
import argparse
import csv
import datetime as dt
import os
import platform
import sys
import time


def now_iso():
    return dt.datetime.now().replace(microsecond=0).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--gemm", type=int, default=4096)
    args = ap.parse_args()

    date = dt.date.today().isoformat()
    out = args.out or os.path.join("results", "pytorch", f"{date}-torch-bench.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []

    def record(test, metric, value, status="ok", notes=""):
        rows.append({
            "start": now_iso(), "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None) or "", "device": dev_name,
            "test": test, "metric": metric, "value": value, "status": status, "notes": notes,
        })
        print(f"  {test:<24} {metric:<8} {value if value is not None else '':>10} {status} {notes}")

    print(f"python {platform.python_version()} on {platform.platform()}")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"import torch failed: {e}")
        sys.exit(1)
    print(f"torch {torch.__version__}, hip {getattr(torch.version, 'hip', None)}, cuda {torch.version.cuda}")

    dev_name = "none"
    avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {avail}")
    if not avail:
        record("device", "available", 0, "fail", "torch.cuda.is_available() is False")
        write(out, rows)
        sys.exit(2)
    dev_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    print(f"device 0: {dev_name} ({arch}), {props.total_memory / 2**30:.1f} GiB, {props.multi_processor_count} CUs")
    record("device", "available", 1, "ok", f"{arch}, {props.total_memory / 2**20:.0f} MiB")

    dev = torch.device("cuda:0")

    def timed(fn, reps):
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    # 1. correctness: matmul vs CPU
    try:
        a = torch.randn(512, 512); b = torch.randn(512, 512)
        ref = a @ b
        got = (a.to(dev) @ b.to(dev)).cpu()
        diff = (ref - got).abs().max().item()
        record("matmul-vs-cpu", "maxdiff", round(diff, 6), "ok" if diff < 1e-2 else "fail", "fp32 512x512")
    except Exception as e:  # noqa: BLE001
        record("matmul-vs-cpu", "maxdiff", None, "fail", repr(e))

    # 2. backward
    try:
        x = torch.randn(64, 128, device=dev, requires_grad=True)
        y = (x * x).sum(); y.backward()
        ok = torch.allclose(x.grad, 2 * x)
        record("backward", "ok", int(ok), "ok" if ok else "fail", "d/dx sum(x^2) == 2x")
    except Exception as e:  # noqa: BLE001
        record("backward", "ok", None, "fail", repr(e))

    # 3. GEMM throughput
    n = args.gemm
    for dtype, label in ((torch.float16, "fp16"), (torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        try:
            a = torch.randn(n, n, device=dev, dtype=dtype); b = torch.randn(n, n, device=dev, dtype=dtype)
            s = timed(lambda: a @ b, args.reps)
            tflops = 2 * n**3 / s / 1e12
            record(f"gemm-{n}-{label}", "TFLOPS", round(tflops, 2), "ok", f"{s*1e3:.1f} ms per matmul, {args.reps} reps")
        except Exception as e:  # noqa: BLE001
            record(f"gemm-{n}-{label}", "TFLOPS", None, "fail", repr(e))

    # 4. conv2d (SD/ResNet-shaped)
    try:
        conv = torch.nn.Conv2d(128, 128, 3, padding=1).to(dev).half()
        x = torch.randn(1, 128, 256, 256, device=dev, dtype=torch.float16)
        with torch.no_grad():
            s = timed(lambda: conv(x), args.reps)
        record("conv3x3-128ch-256px-fp16", "ms", round(s * 1e3, 2), "ok", f"{args.reps} reps, no grad")
    except Exception as e:  # noqa: BLE001
        record("conv3x3-128ch-256px-fp16", "ms", None, "fail", repr(e))

    # 5. training step: tiny MLP, fp32, Adam
    try:
        model = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 1024)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(256, 1024, device=dev); tgt = torch.randn(256, 1024, device=dev)
        def step():
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), tgt)
            loss.backward(); opt.step()
        s = timed(step, args.reps)
        record("train-step-mlp-fp32", "ms", round(s * 1e3, 2), "ok", f"batch 256, Adam, {args.reps} reps")
    except Exception as e:  # noqa: BLE001
        record("train-step-mlp-fp32", "ms", None, "fail", repr(e))

    # 6. torchvision, if present and importable: ResNet-18 inference and a training step
    try:
        import torchvision
        from torchvision.models import resnet18
        tv = torchvision.__version__
        m = resnet18().to(dev).half().eval()
        x = torch.randn(8, 3, 224, 224, device=dev, dtype=torch.float16)
        with torch.no_grad():
            s = timed(lambda: m(x), args.reps)
        record("resnet18-fwd-fp16-b8", "ms", round(s * 1e3, 2), "ok", f"torchvision {tv}")
        m = m.float().train(); x = x.float()
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        y = torch.randint(0, 1000, (8,), device=dev)
        def step():
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(m(x), y).backward(); opt.step()
        s = timed(step, args.reps)
        record("resnet18-train-fp32-b8", "ms", round(s * 1e3, 2), "ok", f"torchvision {tv}, SGD")
        from torchvision.ops import nms
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device=dev)
        keep = nms(boxes, torch.tensor([0.9, 0.8, 0.7], device=dev), 0.5).tolist()
        record("torchvision-nms", "ok", int(keep == [0, 2]), "ok" if keep == [0, 2] else "fail", f"custom op on device, keep={keep}")
    except ImportError:
        record("torchvision", "ok", None, "skip", "torchvision not installed")
    except Exception as e:  # noqa: BLE001
        record("torchvision", "ok", None, "fail", repr(e)[:200])

    # 7. peak memory
    try:
        record("peak-allocated", "MB", round(torch.cuda.max_memory_allocated() / 2**20), "ok")
    except Exception as e:  # noqa: BLE001
        record("peak-allocated", "MB", None, "fail", repr(e))

    write(out, rows)


def write(out, rows):
    if not rows:
        return
    new = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
