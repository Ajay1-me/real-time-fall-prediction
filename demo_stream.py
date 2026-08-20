"""Live demo: streams one fall recording and one ADL recording against the
running FastAPI service (POST /predict per window), reproducing the
notebook's lead-time and false-alarm-rate metrics -- but through the
deployed HTTP API instead of calling the model directly.

Recordings are restricted to subjects held out in the test split, so the
demo is provably run against data the model has not seen during training.

Start the service first:
    python -m uvicorn service.main:app
Then run this script:
    python demo_stream.py
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import requests

from main import load_and_split
from src.inference import accel_magnitude

WINDOW_SIZE = 200
STEP = 50
FS = 200
THRESHOLD = 0.90
DEBOUNCE_K = 2


def predict(base_url: str, window: np.ndarray) -> float:
    resp = requests.post(f"{base_url}/predict", json={"window": window.tolist()})
    resp.raise_for_status()
    return resp.json()["prob_fall"]


def stream_fall(base_url: str, raw_signal: np.ndarray) -> Tuple[float, Optional[float]]:
    consec = 0
    first_alert_time: Optional[float] = None

    for start in range(0, len(raw_signal) - WINDOW_SIZE, STEP):
        window = raw_signal[start:start + WINDOW_SIZE]
        prob = predict(base_url, window)
        t_center = (start + WINDOW_SIZE / 2) / FS

        if prob >= THRESHOLD:
            consec += 1
            if consec >= DEBOUNCE_K and first_alert_time is None:
                first_alert_time = t_center
                print(f"  ALERT at t={t_center:.2f}s (P(fall)={prob:.2f})")
        else:
            consec = 0

    impact_time = int(np.argmax(accel_magnitude(raw_signal))) / FS
    print(f"Impact time ~ {impact_time:.2f}s")
    if first_alert_time is None:
        print("No alert triggered.")
    else:
        print(f"First alert ~ {first_alert_time:.2f}s | Lead time = {impact_time - first_alert_time:.2f}s")

    return impact_time, first_alert_time


def stream_adl(base_url: str, raw_signal: np.ndarray) -> Tuple[int, float]:
    consec, alerts = 0, 0

    for start in range(0, len(raw_signal) - WINDOW_SIZE, STEP):
        window = raw_signal[start:start + WINDOW_SIZE]
        prob = predict(base_url, window)

        if prob >= THRESHOLD:
            consec += 1
            if consec == DEBOUNCE_K:
                alerts += 1
        else:
            consec = 0

    duration_s = len(raw_signal) / FS
    per_min = alerts / (duration_s / 60 + 1e-9)
    print(f"ADL duration: {duration_s:.1f}s | Alerts: {alerts} | Alerts/min: {per_min:.2f}")
    return alerts, per_min


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream recordings against the running fall prediction service")
    parser.add_argument("--data-root", default="SisFall_dataset")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fall-index", type=int, default=0, help="Which fall recording to use")
    parser.add_argument("--adl-index", type=int, default=0, help="Which ADL recording to use")
    args = parser.parse_args()

    try:
        requests.get(f"{args.base_url}/health", timeout=2).raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise SystemExit(
            f"Could not reach the service at {args.base_url} -- is it running?\n"
            f"  python -m uvicorn service.main:app\n{exc}"
        )

    _, _, _, signals, labels, meta, subjects_by_split = load_and_split(args.data_root)
    labels = np.array(labels)
    is_test_subject = meta["subject"].isin(subjects_by_split["test"]).values

    fall_idx = np.where((labels == 1) & is_test_subject)[0][args.fall_index]
    adl_idx = np.where((labels == 0) & is_test_subject)[0][args.adl_index]

    print(f"\n=== Streaming FALL recording ({meta.iloc[fall_idx]['file']}) against {args.base_url} ===")
    stream_fall(args.base_url, signals[fall_idx].astype(np.float32))

    print(f"\n=== Streaming ADL recording ({meta.iloc[adl_idx]['file']}) against {args.base_url} ===")
    stream_adl(args.base_url, signals[adl_idx].astype(np.float32))


if __name__ == "__main__":
    main()
