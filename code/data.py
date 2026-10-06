"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    train_data = np.load(f"{processed_dir}/train.npz")
    eval_data = np.load(f"{processed_dir}/eval.npz")
    
    X_train_full, y_train_full = train_data["X"], train_data["y"]
    X_eval, y_eval, eval_row_id = eval_data["X"], eval_data["y"], eval_data["row_id"]
    
    assert X_train_full.dtype == np.float32 and X_train_full.shape[1] == 54
    assert y_train_full.dtype == np.int64
    assert X_eval.dtype == np.float32 and X_eval.shape[1] == 54
    assert y_eval.dtype == np.int64
    
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn."""
    return train_test_split(X, y, test_size=val_fraction, stratify=y, random_state=seed)


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val)."""
    X_num = X_tr[:, :N_NUMERIC]
    mean = np.mean(X_num, axis=0)
    std = np.std(X_num, axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_out = X.copy()
    X_out[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std
    return X_out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader)."""
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    
    X_tr, X_val, y_tr, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)
    
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)
    
    # Majority class baseline on val
    unique_classes, counts = np.unique(y_val, return_counts=True)
    majority_class_idx = np.argmax(counts)
    majority_acc = counts[majority_class_idx] / len(y_val)
    print(f"Train size: {len(X_tr)}, Val size: {len(X_val)}, Eval size: {len(X_eval)}")
    print(f"Majority class accuracy on val: {majority_acc:.4f}")
    
    return {
        "X_tr": torch.tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)
        
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield X[idx], y[idx]
