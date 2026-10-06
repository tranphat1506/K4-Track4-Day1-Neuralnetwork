"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`."""
from __future__ import annotations

import time
import random
import copy

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",
    optimizer="sgd_momentum",
    lr=0.1,  # Set a reasonable default to not be None
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,
    precision="fp32",
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i+batch_size])
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    preds = []
    
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name, reduction="sum")
        total_loss += loss.item()
        preds.append(logits.argmax(dim=1))
        
    preds = torch.cat(preds)
    acc = (preds == y).float().mean().item()
    
    # confusion matrix
    cm = np.zeros((7, 7), dtype=int)
    y_np = y.cpu().numpy()
    p_np = preds.cpu().numpy()
    for yt, yp in zip(y_np, p_np):
        cm[yt, yp] += 1
        
    macro_f1 = macro_f1_from_confusion(cm)
    avg_loss = total_loss / len(X)
    
    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=7).float()
        return F.mse_loss(logits, y_onehot, reduction=reduction)
    raise ValueError(f"Unknown loss {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])
    device = data["X_tr"].device
    
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[cfg["hidden"]]
    
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], momentum=cfg["momentum"]
    )
    
    scaler = None
    if cfg["precision"] == "fp16":
        # using amp
        try:
            scaler = torch.amp.GradScaler("cuda")
        except:
            scaler = torch.cuda.amp.GradScaler() # Fallback for older pytorch
            
    step0_loss = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])["loss"]
    
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
               "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []}
    
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    diverged = False
    
    # 50k subset for train loss eval
    train_subset_size = min(50000, len(data["X_tr"]))
    perm = torch.randperm(len(data["X_tr"]), device=device)[:train_subset_size]
    X_tr_eval = data["X_tr"][perm]
    y_tr_eval = data["y_tr"][perm]
    
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg["seed"])
    
    for epoch in range(1, cfg["epochs"] + 1):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start_time = time.time()
        
        model.train()
        epoch_grad_norms = []
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator=generator):
            optimizer.zero_grad(set_to_none=True)
            
            if cfg["precision"] == "fp32":
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
                loss.backward()
                
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                optimizer.step()
            else:
                dtype = torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu", dtype=dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
                
                if scaler:
                    scaler.scale(loss).backward()
                    if cfg["clip_norm"] is not None:
                        scaler.unscale_(optimizer)
                    gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                    optimizer.step()
            
            epoch_grad_norms.append(gn)
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break
                
        if diverged:
            break
            
        if torch.cuda.is_available(): torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        
        avg_gn = np.mean(epoch_grad_norms)
        
        train_res = evaluate(model, X_tr_eval, y_tr_eval, cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)
        
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc": history["val_acc"][best_epoch-1] if best_epoch > 0 else None,
        "val_macro_f1": history["val_macro_f1"][best_epoch-1] if best_epoch > 0 else None,
        "time_per_epoch_s": np.mean(history["epoch_time_s"]) if history["epoch_time_s"] else None,
        "peak_mem_MB": torch.cuda.max_memory_allocated(device)/1024/1024 if torch.cuda.is_available() else 0.0,
        "diverged": diverged
    }
    
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    import pandas as pd
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    df.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"Đã lưu kết quả tại {pred_path}. Chạy thủ công: python scripts/evaluate.py --pred {pred_path}")
