"""Validates candidate decision thresholds against the real streaming metrics
that matter -- ADL false alarms per minute and fall detection lead time --
across every held-out test-subject recording, not a single anecdote.

threshold_tuning.py's window-level FPR is a fast proxy; this is the real
check: it replays all 639 test-subject recordings (225 fall, 414 ADL) with
debounced alerting, exactly like demo_stream.py does for one recording at a
time, but computes each recording's per-window probabilities once (batched)
and sweeps every candidate threshold against that cached array -- so this
runs one model pass per recording, not one pass per (recording, threshold)
pair.
"""

import argparse
from typing import List

import numpy as np
import torch
from torch.nn import Module

from main import get_device, load_and_split
from src.inference import accel_magnitude, compute_window_probs, debounced_alerts
from src.model import FallDetectorCNN

DEBOUNCE_K = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate candidate thresholds against streaming metrics")
    parser.add_argument("--data-root", default="SisFall_dataset")
    parser.add_argument("--checkpoint", default="best_fall_detector_cnn.pt")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    )
    args = parser.parse_args()

    device = get_device()
    _, mean, std, signals, labels, meta, subjects_by_split = load_and_split(args.data_root)
    labels = np.array(labels)
    is_test_subject = meta["subject"].isin(subjects_by_split["test"]).values

    model: Module = FallDetectorCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    # Compute per-window probabilities once per recording (the expensive part).
    fall_records = []   # (times, probs, impact_time)
    adl_records = []    # (times, probs, duration_s)

    test_idx = np.where(is_test_subject)[0]
    for i, idx in enumerate(test_idx):
        raw_signal = signals[idx].astype(np.float32)
        times, probs = compute_window_probs(model, raw_signal, mean, std, device)
        if len(times) == 0:
            continue
        if labels[idx] == 1:
            impact_time = int(np.argmax(accel_magnitude(raw_signal))) / 200
            fall_records.append((times, probs, impact_time))
        else:
            duration_s = len(raw_signal) / 200
            adl_records.append((times, probs, duration_s))
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(test_idx)} recordings...")

    print(f"\nProcessed {len(fall_records)} fall and {len(adl_records)} ADL held-out recordings.\n")

    header = f"{'Threshold':>10}{'Falls Detected':>16}{'Avg Lead Time':>16}{'ADL Alerts/min':>16}"
    print(header)
    print("-" * len(header))

    for t in args.thresholds:
        # Fall side: detection rate + average lead time among detections.
        detected = 0
        lead_times: List[float] = []
        for times, probs, impact_time in fall_records:
            alerts = debounced_alerts(times, probs, t, DEBOUNCE_K)
            if alerts:
                detected += 1
                lead_times.append(impact_time - alerts[0])
        detect_rate = detected / len(fall_records) if fall_records else 0.0
        avg_lead = float(np.mean(lead_times)) if lead_times else float("nan")

        # ADL side: aggregate alerts and duration across all recordings for one robust rate.
        total_alerts = 0
        total_duration = 0.0
        for times, probs, duration_s in adl_records:
            total_alerts += len(debounced_alerts(times, probs, t, DEBOUNCE_K))
            total_duration += duration_s
        alerts_per_min = total_alerts / (total_duration / 60 + 1e-9)

        print(f"{t:>10.2f}{detect_rate:>15.1%}{avg_lead:>15.2f}s{alerts_per_min:>15.2f}/min")


if __name__ == "__main__":
    main()
