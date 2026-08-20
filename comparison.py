"""Compares FallDetectorCNN, FallDetectorTCN, Random Forest, and Logistic
Regression on the SisFall test split, reporting F1, precision, recall, and
ROC-AUC for each. Requires the CNN and TCN checkpoints to already exist
(python main.py train --model cnn / --model tcn); the two sklearn baselines
are trained inline since they're cheap.
"""

import argparse
from typing import Dict, List

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from main import CHANNEL_SUBSETS, get_device, load_and_split
from src.data import SisFallWindows
from src.inference import predict_proba
from src.model import FallDetectorCNN, FallDetectorTCN

CHANNEL_NAMES_9 = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z", "acc2_x", "acc2_y", "acc2_z"]


def stats_features(X: np.ndarray) -> np.ndarray:
    """Per-window mean/std/min/max per channel -> (N, 4*C) feature vector for the sklearn baselines."""
    mu, sd = X.mean(axis=1), X.std(axis=1)
    mn, mx = X.min(axis=1), X.max(axis=1)
    return np.concatenate([mu, sd, mn, mx], axis=1)


def feature_names(channel_names: List[str]) -> List[str]:
    return (
        [f"{c}_mean" for c in channel_names]
        + [f"{c}_std" for c in channel_names]
        + [f"{c}_min" for c in channel_names]
        + [f"{c}_max" for c in channel_names]
    )


def print_acc2_importance_ranks(rf: RandomForestClassifier) -> None:
    """Cross-check for the 6-channel variant: are the acc2_xyz features being
    dropped actually low-importance in the 9-channel Random Forest (consistent
    with a small accuracy cost), or high-importance (predicting a bigger hit)?"""
    names = feature_names(CHANNEL_NAMES_9)
    importances = dict(zip(names, rf.feature_importances_))
    ranking = sorted(names, key=lambda n: importances[n], reverse=True)

    print(f"\nRandom Forest feature importance cross-check ({len(names)} total features; "
          f"acc2_* is what the 6-channel variant drops):")
    for name in names:
        if name.startswith("acc2_"):
            rank = ranking.index(name) + 1
            print(f"  {name:<12} importance={importances[name]:.4f}  rank={rank}/{len(names)}")


def torch_model_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    probs, y_true = predict_proba(model, loader, device)
    y_pred = (probs >= 0.5).astype(int)
    return {
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, probs),
    }


def sklearn_model_metrics(clf, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    return {
        "F1": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, probs),
    }


def print_table(results: Dict[str, Dict[str, float]]) -> None:
    header = f"{'Model':<22}{'F1':>8}{'Precision':>12}{'Recall':>10}{'ROC-AUC':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<22}{m['F1']:>8.3f}{m['Precision']:>12.3f}{m['Recall']:>10.3f}{m['ROC-AUC']:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN vs TCN vs Random Forest vs Logistic Regression")
    parser.add_argument("--data-root", default="SisFall_dataset")
    parser.add_argument("--cnn-checkpoint", default="best_fall_detector_cnn.pt")
    parser.add_argument("--cnn6-checkpoint", default="best_fall_detector_cnn_6ch.pt")
    parser.add_argument("--tcn-checkpoint", default="best_fall_detector_tcn.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    splits, _, _, _, _, _, _ = load_and_split(args.data_root)
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    test_loader = DataLoader(SisFallWindows(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    results: Dict[str, Dict[str, float]] = {}

    cnn = FallDetectorCNN().to(device)
    cnn.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device, weights_only=True))
    results["CNN (9-channel)"] = torch_model_metrics(cnn, test_loader, device)

    splits_6ch, _, _, _, _, _, _ = load_and_split(args.data_root, channels=CHANNEL_SUBSETS[6])
    X_test_6ch, y_test_6ch = splits_6ch["test"]
    test_loader_6ch = DataLoader(SisFallWindows(X_test_6ch, y_test_6ch), batch_size=args.batch_size, shuffle=False)

    cnn6 = FallDetectorCNN(in_channels=6).to(device)
    cnn6.load_state_dict(torch.load(args.cnn6_checkpoint, map_location=device, weights_only=True))
    results["CNN (6-channel, phone)"] = torch_model_metrics(cnn6, test_loader_6ch, device)

    tcn = FallDetectorTCN().to(device)
    tcn.load_state_dict(torch.load(args.tcn_checkpoint, map_location=device, weights_only=True))
    results["TCN"] = torch_model_metrics(tcn, test_loader, device)

    Xtr_f, Xte_f = stats_features(X_train), stats_features(X_test)

    lr = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    lr.fit(Xtr_f, y_train)
    results["Logistic Regression"] = sklearn_model_metrics(lr, Xte_f, y_test)

    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    rf.fit(Xtr_f, y_train)
    results["Random Forest"] = sklearn_model_metrics(rf, Xte_f, y_test)

    print()
    print_table(results)
    print_acc2_importance_ranks(rf)


if __name__ == "__main__":
    main()
