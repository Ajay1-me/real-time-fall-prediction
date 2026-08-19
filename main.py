"""Command-line entry point for the fall prediction pipeline.

Usage:
    python main.py train
    python main.py evaluate
    python main.py simulate --kind fall --index 0
    python main.py simulate --kind adl --index 0
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from src.data import SisFallWindows, create_windows_with_record_idx, load_sisfall_dataset, save_norm_stats
from src.inference import simulate_adl_false_alarms, simulate_realtime_with_context
from src.model import FallDetectorCNN, FallDetectorTCN, get_device
from src.train import evaluate as evaluate_model
from src.train import train_model

MODEL_CLASSES = {"cnn": FallDetectorCNN, "tcn": FallDetectorTCN}


def resolve_checkpoint(args: argparse.Namespace) -> str:
    return args.checkpoint or f"best_fall_detector_{args.model}.pt"


def load_and_split(data_root: str, window_size: int = 200, overlap: float = 0.5):
    """Loads SisFall, windows it, and reproduces the notebook's subject-wise
    train/val/test split (random_state=42 makes this deterministic across runs)."""
    signals, labels, meta = load_sisfall_dataset(data_root)
    X_all, y_all, rec_idx = create_windows_with_record_idx(signals, labels, window_size, overlap)
    groups = meta["subject"].values[rec_idx]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(X_all, y_all, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel, test_rel = next(gss2.split(X_all[temp_idx], y_all[temp_idx], groups=groups[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    splits = {
        "train": (X_all[train_idx].astype(np.float32), y_all[train_idx].astype(np.int64)),
        "val": (X_all[val_idx].astype(np.float32), y_all[val_idx].astype(np.int64)),
        "test": (X_all[test_idx].astype(np.float32), y_all[test_idx].astype(np.int64)),
    }

    mean = splits["train"][0].mean(axis=(0, 1), keepdims=True)
    std = splits["train"][0].std(axis=(0, 1), keepdims=True) + 1e-6
    for key, (X, y) in splits.items():
        splits[key] = ((X - mean) / std, y)

    subjects_by_split = {
        "train": set(groups[train_idx]),
        "val": set(groups[val_idx]),
        "test": set(groups[test_idx]),
    }

    return splits, mean, std, signals, labels, meta, subjects_by_split


def cmd_train(args: argparse.Namespace) -> None:
    device = get_device()
    checkpoint = resolve_checkpoint(args)
    splits, mean, std, _, _, _, _ = load_and_split(args.data_root)
    (X_train, y_train), (X_val, y_val) = splits["train"], splits["val"]

    train_loader = DataLoader(SisFallWindows(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SisFallWindows(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = MODEL_CLASSES[args.model]().to(device)
    class_weights = torch.tensor(
        len(y_train) / (2 * np.bincount(y_train).astype(np.float32)), dtype=torch.float32, device=device
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=args.epochs, patience=args.patience, checkpoint_path=checkpoint,
    )
    save_norm_stats(mean, std, args.norm_stats)
    print(f"Saved best checkpoint to {checkpoint} and normalization stats to {args.norm_stats}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    device = get_device()
    checkpoint = resolve_checkpoint(args)
    splits, _, _, _, _, _, _ = load_and_split(args.data_root)
    X_test, y_test = splits["test"]
    test_loader = DataLoader(SisFallWindows(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = MODEL_CLASSES[args.model]().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    criterion = torch.nn.CrossEntropyLoss()

    _, f1, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    print(f"Test F1 (fall vs no-fall): {f1:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=["ADL", "Fall"]))


def cmd_simulate(args: argparse.Namespace) -> None:
    device = get_device()
    checkpoint = resolve_checkpoint(args)
    _, mean, std, signals, labels, _, _ = load_and_split(args.data_root)

    model = MODEL_CLASSES[args.model]().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    target_label = 1 if args.kind == "fall" else 0
    matching = np.where(np.array(labels) == target_label)[0]
    idx = matching[args.index]

    if args.kind == "fall":
        simulate_realtime_with_context(model, signals[idx], mean, std, device, threshold=args.threshold)
    else:
        simulate_adl_false_alarms(model, signals[idx], mean, std, device, threshold=args.threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time fall prediction pipeline")
    parser.add_argument("--data-root", default="SisFall_dataset")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Defaults to best_fall_detector_<model>.pt based on --model",
    )
    parser.set_defaults(batch_size=128, model="cnn")  # so bare `python main.py` (-> cmd_evaluate) has these set
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train a model and save the best checkpoint")
    p_train.add_argument("--model", choices=MODEL_CLASSES.keys(), default="cnn")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--patience", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--norm-stats", default="norm_stats.npz")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a checkpoint on the held-out test split")
    p_eval.add_argument("--model", choices=MODEL_CLASSES.keys(), default="cnn")
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.set_defaults(func=cmd_evaluate)

    p_sim = sub.add_parser("simulate", help="Stream one recording through the model")
    p_sim.add_argument("--model", choices=MODEL_CLASSES.keys(), default="cnn")
    p_sim.add_argument("--kind", choices=["fall", "adl"], default="fall")
    p_sim.add_argument("--index", type=int, default=0, help="Which recording of that kind to use")
    p_sim.add_argument("--threshold", type=float, default=0.7)
    p_sim.set_defaults(func=cmd_simulate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        # No subcommand given (e.g. hitting "Run" in an IDE) -> demo the
        # existing checkpoint against the test split instead of erroring out.
        print("No subcommand given, defaulting to `evaluate`.\n")
        cmd_evaluate(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
