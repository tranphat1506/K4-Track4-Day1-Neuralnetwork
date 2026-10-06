import sys
sys.path.append('code')
import torch
from data import prepare_data
from train import run_experiment, DEFAULT_CFG

print("Loading data...")
data = prepare_data("cpu", processed_dir="data/processed")
cfg = DEFAULT_CFG.copy()
cfg["epochs"] = 1
cfg["batch"] = 8192 # Large batch for speed
print("Running experiment...")
res = run_experiment(cfg, data)
print(f"Done. Final Val Loss: {res['summary']['final_val_loss']:.4f}, Val Acc: {res['summary']['val_acc']:.4f}")
