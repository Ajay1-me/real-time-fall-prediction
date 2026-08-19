"""Benchmarks per-window CPU inference latency for FallDetectorCNN and checks
whether it comfortably fits inside the streaming step interval (window_size=200,
step=50 at 200Hz -> a new window arrives every 250ms), which is the actual
real-time constraint the model has to keep up with.
"""

import argparse
import time
from typing import List

import numpy as np
import torch

from main import load_and_split
from src.model import FallDetectorCNN

STEP_SIZE = 50
FS = 200
STEP_INTERVAL_MS = (STEP_SIZE / FS) * 1000  # 250.0


def benchmark(model: torch.nn.Module, X: np.ndarray, device: torch.device, warmup: int) -> List[float]:
    model.eval()
    windows = torch.from_numpy(X).permute(0, 2, 1).to(device)  # (N, C, T)

    with torch.no_grad():
        for i in range(warmup):
            model(windows[i % len(windows): i % len(windows) + 1])

    latencies_ms = []
    with torch.no_grad():
        for i in range(len(windows)):
            x = windows[i:i + 1]  # batch size 1, matching a real streaming window arriving one at a time
            start = time.perf_counter()
            model(x)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

    return latencies_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark per-window CPU inference latency for the CNN")
    parser.add_argument("--data-root", default="SisFall_dataset")
    parser.add_argument("--checkpoint", default="best_fall_detector_cnn.pt")
    parser.add_argument("--num-windows", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cpu")  # CPU is the worst-case deployment target for a wearable/edge device
    splits, _, _, _, _ = load_and_split(args.data_root)
    X_test, _ = splits["test"]
    n = min(args.num_windows, len(X_test))

    model = FallDetectorCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    latencies_ms = benchmark(model, X_test[:n], device, args.warmup)
    avg = float(np.mean(latencies_ms))
    p95 = float(np.percentile(latencies_ms, 95))

    print(f"CPU inference latency over {n} windows (batch size 1):")
    print(f"  avg: {avg:.3f} ms")
    print(f"  p95: {p95:.3f} ms")
    print(f"Streaming step interval (window_size=200, step=50, fs=200Hz): {STEP_INTERVAL_MS:.1f} ms")

    if p95 < STEP_INTERVAL_MS:
        print(f"Fits comfortably: p95 latency is {STEP_INTERVAL_MS / p95:.1f}x faster than the step interval.")
    else:
        print("WARNING: p95 latency exceeds the streaming step interval; the model would fall behind real-time input.")


if __name__ == "__main__":
    main()
