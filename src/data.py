"""SisFall dataset loading, windowing, and the PyTorch Dataset wrapper."""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_sisfall_dataset(root: str = "SisFall_dataset") -> Tuple[List[np.ndarray], np.ndarray, pd.DataFrame]:
    all_signals, all_labels, meta = [], [], []

    # Bit-to-physical unit conversion factors, per the SisFall sensor datasheets
    g_factor_ADXL345 = (2 * 16) / (2 ** 13)
    g_factor_MMA8451Q = (2 * 8) / (2 ** 14)
    deg_factor_ITG3200 = (2 * 2000) / (2 ** 16)

    for subj_folder in sorted(os.listdir(root)):
        subj_path = os.path.join(root, subj_folder)
        if not os.path.isdir(subj_path):
            continue

        for file in sorted(os.listdir(subj_path)):
            if not file.endswith(".txt"):
                continue

            # Label from filename prefix (Fxx -> fall=1, Dxx -> ADL=0)
            label = 1 if file.startswith("F") else 0

            df = pd.read_csv(os.path.join(subj_path, file),
                              sep='[;,\\s]+', engine='python', header=None)
            df = df.dropna(axis=1, how='all')
            if df.shape[1] < 9:  # skip corrupted
                continue
            df = df.iloc[:, :9].astype(np.float32)

            df.iloc[:, 0:3] *= g_factor_ADXL345
            df.iloc[:, 3:6] *= deg_factor_ITG3200
            df.iloc[:, 6:9] *= g_factor_MMA8451Q

            all_signals.append(df.values)
            all_labels.append(label)
            meta.append({
                "file": file,
                "subject": subj_folder,
                "samples": len(df),
                "label": label,
            })

    print(f"Loaded {len(all_signals)} recordings "
          f"({sum(all_labels)} falls, {len(all_labels) - sum(all_labels)} ADL)")
    return all_signals, np.array(all_labels), pd.DataFrame(meta)


def create_windows_with_record_idx(
    signals: List[np.ndarray],
    labels: np.ndarray,
    window_size: int = 200,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows each recording and tracks which recording each window came from,
    so windows can later be grouped by subject for a leakage-free split."""
    X_list, y_list, rec_idx = [], [], []
    step = int(window_size * (1 - overlap))
    for i, (sig, lab) in enumerate(zip(signals, labels)):
        for start in range(0, len(sig) - window_size + 1, step):
            X_list.append(sig[start:start + window_size])
            y_list.append(lab)
            rec_idx.append(i)
    return np.array(X_list), np.array(y_list), np.array(rec_idx)


class SisFallWindows(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # X: (N, T, C) -> (N, C, T) for Conv1d
        self.X = torch.from_numpy(X).permute(0, 2, 1).contiguous()
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
