"""plots.py — PSEUDO-CODE."""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG."""
    cfg = result["cfg"]
    history = result["history"]
    epochs = history["epoch"]
    best_epoch = result["summary"]["best_epoch"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    title = f"{cfg['exp_id']} | opt:{cfg['optimizer']} lr:{cfg['lr']} batch:{cfg['batch']} init:{cfg['init']}"
    fig.suptitle(title)
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f"Best ({best_epoch})")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    
    # Acc / Macro F1
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1")
    axes[1].axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f"Best ({best_epoch})")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    
    # Grad Norm
    axes[2].plot(epochs, history["grad_norm"], label="Grad Norm (pre-clip)", color="orange")
    if cfg["clip_norm"]:
        axes[2].axhline(cfg["clip_norm"], color='red', linestyle=':', label=f"Clip {cfg['clip_norm']}")
    axes[2].set_title("Gradient Norm")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for res in results:
        cfg = res["cfg"]
        history = res["history"]
        ax.plot(history["epoch"], history[metric], label=cfg["exp_id"])
        
    ax.set_title(title if title else f"Comparison: {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
