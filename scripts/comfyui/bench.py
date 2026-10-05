"""Headless ComfyUI benchmark: starts the server, submits a text-to-image workflow per
(config, rep), times the sampler and VAE stages over the websocket, and writes a CSV.

Usage (repo root):
    bin\\comfyui\\.venv\\Scripts\\python.exe scripts\\comfyui\\bench.py [--reps 3] [--server-args "--use-split-cross-attention"]

Per run it records: steps/s from the sampler's tqdm line in the server log, sampling
seconds (KSampler start to VAEDecode start, includes moving the model to VRAM on the
first run), decode seconds, and total prompt seconds. The server is started with
--cache-none so every rep re-executes; model weights stay in RAM between reps.
"""
import argparse
import asyncio
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import uuid

import aiohttp  # a ComfyUI dependency, so always present in its venv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CONFIGS = [
    {"label": "512-euler_a-20", "width": 512, "height": 512, "steps": 20, "sampler": "euler_ancestral"},
    {"label": "512-dpmpp2m-20", "width": 512, "height": 512, "steps": 20, "sampler": "dpmpp_2m"},
    {"label": "768-euler_a-20", "width": 768, "height": 768, "steps": 20, "sampler": "euler_ancestral"},
]

PROMPT = "a lighthouse on a rocky coast at sunset, oil painting"
NEGATIVE = "blurry, low quality"
TQDM = re.compile(r"(\d+)/(\d+) \[[\d:]+<[\d:]+,\s*([\d.]+)(it/s|s/it)\]")


def workflow(ckpt, cfg, seed, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": cfg["width"], "height": cfg["height"], "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": cfg["steps"], "cfg": 7.0,
              "sampler_name": cfg["sampler"], "scheduler": "normal", "denoise": 1.0,
              "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["6", 0]}},
    }


async def run_prompt(base, client_id, wf):
    """Submit wf and return per-node start times plus end time (perf_counter seconds)."""
    starts, end = {}, None
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"ws://{base}/ws?clientId={client_id}", max_msg_size=0) as ws:
            async with s.post(f"http://{base}/prompt", json={"prompt": wf, "client_id": client_id}) as r:
                body = await r.json()
                if r.status != 200:
                    raise RuntimeError(f"/prompt {r.status}: {json.dumps(body)[:500]}")
                pid = body["prompt_id"]
            t_submit = time.perf_counter()
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                d = m.get("data", {})
                if d.get("prompt_id") not in (None, pid):
                    continue
                if m["type"] == "execution_start":
                    starts["_start"] = time.perf_counter()
                elif m["type"] == "executing":
                    if d.get("node") is None:
                        end = time.perf_counter(); break
                    starts.setdefault(d["node"], time.perf_counter())
                elif m["type"] in ("execution_error", "execution_interrupted"):
                    raise RuntimeError(f"{m['type']}: {json.dumps(d)[:800]}")
                elif m["type"] == "execution_success":
                    end = time.perf_counter(); break
    starts.setdefault("_start", t_submit)
    return starts, end


def wait_server(base, proc, timeout=180):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://{base}/system_stats", timeout=2) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError("server did not come up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=os.path.join(ROOT, "src", "ComfyUI"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--model-paths", default=os.path.join(ROOT, "bin", "comfyui", "extra_model_paths.yaml"))
    ap.add_argument("--ckpt", default="v1-5-pruned-emaonly-fp16.safetensors")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-suffix", default="", help="appended to config labels, e.g. -split")
    ap.add_argument("--server-args", default="", help="extra ComfyUI flags, e.g. --use-split-cross-attention")
    ap.add_argument("--out", default=None)
    ap.add_argument("--configs", default=None, help="JSON list overriding the defaults")
    args = ap.parse_args()

    configs = json.loads(args.configs) if args.configs else DEFAULT_CONFIGS
    date = dt.date.today().isoformat()
    out = args.out or os.path.join(ROOT, "results", "comfyui", f"{date}-comfyui-bench.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log_path = os.path.join(os.path.dirname(out), f"{date}-comfyui-server{args.label_suffix}.log")
    base = f"127.0.0.1:{args.port}"

    cmd = [args.python, "main.py", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch",
           "--cache-none", "--extra-model-paths-config", args.model_paths,
           "--output-directory", os.path.join(ROOT, "results", "comfyui", "images")] + args.server_args.split()
    print("starting:", " ".join(cmd))
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, cwd=args.comfy, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        stats = wait_server(base, proc)
        dev = stats["devices"][0]
        version = stats["system"].get("comfyui_version", "")
        torch_v = stats["system"].get("pytorch_version", "")
        print(f"ComfyUI {version}, torch {torch_v}, device {dev['name']}, vram {dev['vram_total'] // 2**20} MiB")
        pos = 0
        for cfg in configs:
            for rep in range(1, args.reps + 1):
                label = cfg["label"] + args.label_suffix
                wf = workflow(args.ckpt, cfg, args.seed, f"{label}-r{rep}")
                starts, end = asyncio.run(run_prompt(base, uuid.uuid4().hex, wf))
                # parse the sampler's final tqdm line from the newly written log
                log.flush()
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    f.seek(pos); chunk = f.read(); pos = f.tell()
                its = None
                for m in TQDM.finditer(chunk.replace("\r", "\n")):
                    if m.group(1) == m.group(2) == str(cfg["steps"]):
                        v = float(m.group(3)); its = v if m.group(4) == "it/s" else 1 / v
                sampling = starts.get("6", end) - starts.get("5", starts["_start"])
                decode = starts.get("7", end) - starts.get("6", end)
                total = end - starts["_start"]
                row = {"label": label, "start": dt.datetime.now().replace(microsecond=0).isoformat(),
                       "comfyui": version, "torch": torch_v, "device": dev["name"], "server_args": args.server_args,
                       "width": cfg["width"], "height": cfg["height"], "steps": cfg["steps"], "sampler": cfg["sampler"],
                       "rep": rep, "steps_per_s": its, "sampling_s": round(sampling, 2), "decode_s": round(decode, 2),
                       "total_s": round(total, 2)}
                rows.append(row)
                print(f"  {label:<28} rep {rep}: {its} it/s, sampling {sampling:.2f} s, decode {decode:.2f} s, total {total:.2f} s")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if rows:
        new = not os.path.exists(out)
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"Results: {out}\nServer log: {log_path}")


if __name__ == "__main__":
    main()
