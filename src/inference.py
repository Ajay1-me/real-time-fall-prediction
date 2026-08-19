"""Real-time streaming simulation and prediction utilities."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def accel_magnitude(raw_signal: np.ndarray) -> np.ndarray:
    # raw_signal: (T,9). Uses the acc1 triad (cols 0..2).
    ax, ay, az = raw_signal[:, 0], raw_signal[:, 1], raw_signal[:, 2]
    return np.sqrt(ax * ax + ay * ay + az * az)


def simulate_realtime_with_context(
    model: nn.Module,
    raw_signal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    window_size: int = 200,
    step: int = 50,
    fs: int = 200,
    threshold: float = 0.7,
    debounce_k: int = 2,
    plot: bool = True,
) -> Tuple[List[float], List[float], float, Optional[float], Optional[float]]:
    """Streams a raw recording through the model window-by-window.

    debounce_k requires the threshold to be crossed for K consecutive windows
    before raising an alert, to suppress single-window false positives.
    Impact time is estimated as the peak of raw acceleration magnitude, and
    lead_time = impact_time - first_alert_time quantifies the early-warning gap.
    """
    model.eval()
    raw_signal = raw_signal.astype(np.float32)
    norm_signal = (raw_signal - mean.squeeze(0)) / std.squeeze(0)

    times, probs = [], []
    consec = 0
    first_alert_time = None

    with torch.no_grad():
        for start in range(0, len(norm_signal) - window_size, step):
            end = start + window_size
            window = norm_signal[start:end]
            x = torch.from_numpy(window).unsqueeze(0).permute(0, 2, 1).to(device)

            p = torch.softmax(model(x), dim=1)[0, 1].item()
            t_center = (start + window_size / 2) / fs
            times.append(t_center)
            probs.append(p)

            if p >= threshold:
                consec += 1
                if consec >= debounce_k and first_alert_time is None:
                    first_alert_time = t_center
            else:
                consec = 0

    amag = accel_magnitude(raw_signal)
    impact_idx = int(np.argmax(amag))
    impact_time = impact_idx / fs

    lead_time = None if first_alert_time is None else (impact_time - first_alert_time)

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(times, probs)
        plt.axhline(threshold, linestyle="--")
        plt.axvline(impact_time, linestyle=":", label=f"Impact ~ {impact_time:.2f}s")
        if first_alert_time is not None:
            plt.axvline(first_alert_time, linestyle=":", label=f"First alert ~ {first_alert_time:.2f}s")
        plt.xlabel("Time (s)")
        plt.ylabel("P(fall)")
        plt.title("Real-time P(fall) with impact + first alert markers")
        plt.legend()
        plt.tight_layout()
        plt.show()

        t_raw = np.arange(len(amag)) / fs
        plt.figure(figsize=(10, 3))
        plt.plot(t_raw, amag)
        plt.axvline(impact_time, linestyle=":")
        plt.xlabel("Time (s)")
        plt.ylabel("|acc| (g)")
        plt.title("Acceleration magnitude (impact peak marker)")
        plt.tight_layout()
        plt.show()

    print(f"Impact time ~ {impact_time:.2f}s")
    if first_alert_time is None:
        print("No alert triggered.")
    else:
        print(f"First alert time ~ {first_alert_time:.2f}s  |  Lead time = {lead_time:.2f}s")
    return times, probs, impact_time, first_alert_time, lead_time


def simulate_adl_false_alarms(
    model: nn.Module,
    raw_signal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    window_size: int = 200,
    step: int = 50,
    fs: int = 200,
    threshold: float = 0.7,
    debounce_k: int = 2,
) -> Tuple[int, float]:
    model.eval()
    raw_signal = raw_signal.astype(np.float32)
    norm_signal = (raw_signal - mean.squeeze(0)) / std.squeeze(0)

    consec, alerts = 0, 0
    with torch.no_grad():
        for start in range(0, len(norm_signal) - window_size, step):
            end = start + window_size
            window = norm_signal[start:end]
            x = torch.from_numpy(window).unsqueeze(0).permute(0, 2, 1).to(device)
            p = torch.softmax(model(x), dim=1)[0, 1].item()

            if p >= threshold:
                consec += 1
                if consec == debounce_k:
                    alerts += 1
            else:
                consec = 0

    duration_s = len(raw_signal) / fs
    per_min = alerts / (duration_s / 60 + 1e-9)
    print(f"ADL duration: {duration_s:.1f}s | Alerts: {alerts} | Alerts/min: {per_min:.2f}")
    return alerts, per_min
