"""Sweeps the decision threshold for the 9-channel CNN on the held-out test
split, reporting precision/recall/F1/false-positive-rate at each threshold
and recommending an operating point. Every other script in this repo uses a
fixed threshold (0.5 via argmax in comparison.py, 0.7 as a default elsewhere)
that was never actually chosen based on this tradeoff -- this is that choice,
made explicit.

Note: this FPR is the window-level false-positive rate on the i.i.d. test
split, not the same thing as the streaming "alerts per minute" metric from
demo_stream.py / simulate_adl_false_alarms (which replays whole recordings
with debouncing). It's a fast, useful proxy for picking a candidate
threshold, but the recommended threshold should be validated against the
real streaming metric (see validate_thresholds.py) before treating it as
final -- window-level and streaming metrics can disagree substantially,
since a single successful debounced alert anywhere in a fall recording is
enough, regardless of how many individual windows were misclassified.
"""

import argparse
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from main import get_device, load_and_split
from src.data import SisFallWindows
from src.inference import predict_proba
from src.model import FallDetectorCNN


def sweep(probs: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray) -> List[Dict[str, float]]:
    rows = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append({
            "threshold": float(t),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "fpr": fp / (fp + tn + 1e-9),
        })
    return rows


def recommend_threshold(rows: List[Dict[str, float]], min_recall: float) -> Dict[str, float]:
    """Among thresholds meeting the minimum recall bar, pick the one with the lowest FPR."""
    candidates = [r for r in rows if r["recall"] >= min_recall]
    if not candidates:
        return max(rows, key=lambda r: r["recall"])  # fall back: best recall available, bar unmet
    return min(candidates, key=lambda r: r["fpr"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep decision threshold for the 9-channel CNN")
    parser.add_argument("--data-root", default="SisFall_dataset")
    parser.add_argument("--checkpoint", default="best_fall_detector_cnn.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--min-recall", type=float, default=0.85,
        help="Minimum acceptable fall recall when recommending a threshold",
    )
    args = parser.parse_args()

    device = get_device()
    splits, _, _, _, _, _, _ = load_and_split(args.data_root)
    X_test, y_test = splits["test"]
    test_loader = DataLoader(SisFallWindows(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = FallDetectorCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    probs, y_true = predict_proba(model, test_loader, device)

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = sweep(probs, y_true, thresholds)

    header = f"{'Threshold':>10}{'Precision':>12}{'Recall':>10}{'F1':>8}{'FPR':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['threshold']:>10.2f}{r['precision']:>12.3f}{r['recall']:>10.3f}{r['f1']:>8.3f}{r['fpr']:>10.3f}")

    best = recommend_threshold(rows, args.min_recall)
    print(f"\nRecommended threshold (lowest window-level FPR with recall >= {args.min_recall}): "
          f"{best['threshold']:.2f}  (precision={best['precision']:.3f}, recall={best['recall']:.3f}, "
          f"f1={best['f1']:.3f}, fpr={best['fpr']:.3f})")
    print("This is a window-level estimate -- validate against the real streaming metrics "
          "(validate_thresholds.py) before treating it as the production threshold.")


if __name__ == "__main__":
    main()
