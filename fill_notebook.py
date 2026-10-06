import json

with open("code/lab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

data_cell = [
    "import subprocess\n",
    "import sys\n",
    "import numpy as np\n",
    "import torch\n",
    "from data import prepare_data\n",
    "\n",
    "REPO_ROOT = '..'\n",
    "OUT_DIR = '../submission_123456'\n",
    "import os\n",
    "os.makedirs(f'{OUT_DIR}/figures', exist_ok=True)\n",
    "os.makedirs(f'{OUT_DIR}/results', exist_ok=True)\n",
    "\n",
    "print('Chạy split_data.py...')\n",
    "subprocess.run([sys.executable, f'{REPO_ROOT}/scripts/split_data.py'], check=True)\n",
    "\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "data = prepare_data(device, val_fraction=0.2, seed=42, processed_dir=f'{REPO_ROOT}/data/processed')\n",
    "\n",
    "print(f'X_tr shape: {data[\"X_tr\"].shape}')\n",
    "print(f'X_val shape: {data[\"X_val\"].shape}')\n",
    "print(f'X_eval shape: {data[\"X_eval\"].shape}')\n",
    "\n",
    "val_y = data['y_val'].cpu().numpy()\n",
    "most_frequent_class = np.bincount(val_y).argmax()\n",
    "baseline_acc = np.mean(val_y == most_frequent_class)\n",
    "print(f'Accuracy đoán lớp đa số trên val: {baseline_acc:.4f}')\n"
]

health_cell = [
    "from model import MLP, EXPECTED_PARAMS, count_params\n",
    "import torch.nn as nn\n",
    "\n",
    "model = MLP(hidden=(256, 128), dropout=0.0, init='he').to(device)\n",
    "assert count_params(model) == EXPECTED_PARAMS[(256,128)]\n",
    "print('Số tham số hợp lệ:', count_params(model))\n",
    "\n",
    "dummy_x = torch.randn(8, 54).to(device)\n",
    "logits = model(dummy_x)\n",
    "assert logits.shape == (8, 7)\n",
    "print('Shape đầu ra hợp lệ:', logits.shape)\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "loss = criterion(logits, torch.randint(0, 7, (8,)).to(device))\n",
    "print(f'Loss bước 0 ban đầu: {loss.item():.4f} (Kỳ vọng ~ 1.946)')\n",
    "\n",
    "import torch.optim as optim\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "optimizer = optim.SGD(model.parameters(), lr=0.1)\n",
    "x_mini = data['X_tr'][:20]\n",
    "y_mini = data['y_tr'][:20]\n",
    "losses = []\n",
    "for _ in range(200):\n",
    "    optimizer.zero_grad()\n",
    "    out = model(x_mini)\n",
    "    l = criterion(out, y_mini)\n",
    "    l.backward()\n",
    "    optimizer.step()\n",
    "    losses.append(l.item())\n",
    "\n",
    "plt.plot(losses)\n",
    "plt.title('Overfit 20 samples')\n",
    "plt.show()\n",
    "\n",
    "for name, p in model.named_parameters():\n",
    "    if p.grad is not None:\n",
    "        print(f'{name} grad norm: {p.grad.norm().item():.4f}')\n",
    "        assert p.grad.norm().item() > 0\n"
]

baseline_cell = [
    "from train import DEFAULT_CFG, run_experiment\n",
    "from results_table import save_result\n",
    "from plots import plot_run\n",
    "\n",
    "cfg = DEFAULT_CFG.copy()\n",
    "cfg['lr'] = 0.05\n",
    "cfg['epochs'] = 20\n",
    "\n",
    "val_macros = []\n",
    "val_accs = []\n",
    "for s in [42, 43, 44]:\n",
    "    exp_cfg = {**cfg, 'seed': s, 'exp_id': f'base-s{s}', 'group': 'baseline'}\n",
    "    res = run_experiment(exp_cfg, data)\n",
    "    save_result(res, f'{OUT_DIR}/results')\n",
    "    plot_run(res, f'{OUT_DIR}/figures/{exp_cfg[\"exp_id\"]}.png')\n",
    "    val_macros.append(res['summary']['val_macro_f1'])\n",
    "    val_accs.append(res['summary']['val_acc'])\n",
    "\n",
    "print(f'Val Macro-F1 (Baseline): {np.mean(val_macros):.4f} ± {np.std(val_macros):.4f}')\n",
    "print(f'Val Acc (Baseline): {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}')\n"
]

experiment_cell = [
    "experiments = [\n",
    "    {'exp_id': 'adam-lr1e-3', 'group': 'optimizer', 'description': 'Adam lr 0.001', 'optimizer': 'adam', 'lr': 0.001},\n",
    "    {'exp_id': 'adamw-lr1e-3', 'group': 'optimizer', 'description': 'AdamW lr 0.001', 'optimizer': 'adamw', 'lr': 0.001},\n",
    "    {'exp_id': 'model-wide', 'group': 'architecture', 'description': 'Wide model', 'hidden': (512, 256)},\n",
    "    {'exp_id': 'model-deep', 'group': 'architecture', 'description': 'Deep model', 'hidden': (256, 128, 64)},\n",
    "]\n",
    "\n",
    "for exp in experiments:\n",
    "    print(f'Running {exp[\"exp_id\"]}...')\n",
    "    exp_cfg = {**DEFAULT_CFG, **exp, 'seed': 42}\n",
    "    res = run_experiment(exp_cfg, data)\n",
    "    save_result(res, f'{OUT_DIR}/results')\n",
    "    plot_run(res, f'{OUT_DIR}/figures/{exp_cfg[\"exp_id\"]}.png')\n",
    "    print(f'Done {exp[\"exp_id\"]}: Val F1 = {res[\"summary\"][\"val_macro_f1\"]:.4f}')\n"
]

compare_cell = [
    "from results_table import load_results\n",
    "from plots import plot_compare\n",
    "results_all = load_results(f'{OUT_DIR}/results')\n",
    "opt_results = [r for r in results_all if r['cfg'].get('group') in ['optimizer', 'baseline']]\n",
    "arch_results = [r for r in results_all if r['cfg'].get('group') in ['architecture', 'baseline']]\n",
    "\n",
    "plot_compare(opt_results, 'val_loss', f'{OUT_DIR}/figures/compare_optimizer.png')\n",
    "plot_compare(arch_results, 'val_loss', f'{OUT_DIR}/figures/compare_architecture.png')\n"
]

final_eval_cell = [
    "from train import final_eval\n",
    "best_cfg = {**DEFAULT_CFG, 'exp_id': 'adamw-lr1e-3', 'optimizer': 'adamw', 'lr': 0.001, 'seed': 42}\n",
    "best_res = run_experiment(best_cfg, data)\n",
    "final_eval(best_cfg, best_res, data, f'{OUT_DIR}/predictions_eval.csv')\n",
    "\n",
    "import subprocess\n",
    "subprocess.run([sys.executable, f'{REPO_ROOT}/scripts/evaluate.py', '--pred', f'{OUT_DIR}/predictions_eval.csv', '--out', f'{OUT_DIR}/eval_result.json'], check=True)\n"
]

error_analysis_cell = [
    "import json\n",
    "with open(f'{OUT_DIR}/eval_result.json', 'r') as f:\n",
    "    eval_res = json.load(f)\n",
    "print('Macro F1:', eval_res['macro_f1'])\n",
    "print('Accuracy:', eval_res['accuracy'])\n",
    "print('F1 by class:', eval_res['f1_scores'])\n"
]

write_excel_cell = [
    "from results_table import load_results, to_row, write_xlsx\n",
    "results = load_results(f'{OUT_DIR}/results')\n",
    "rows = [to_row(r, eval_json=f'{OUT_DIR}/eval_result.json' if r['cfg']['exp_id'] == 'adamw-lr1e-3' else None) for r in results]\n",
    "write_xlsx(rows, f'{REPO_ROOT}/templates/experiment_table_template.xlsx', f'{OUT_DIR}/experiments.xlsx')\n",
    "print('Done writing Excel.')\n"
]

code_cells_content = [data_cell, health_cell, baseline_cell, experiment_cell, compare_cell, final_eval_cell, error_analysis_cell, write_excel_cell]

code_idx = 0
# The first code cell is imports, we skip it.
found_first_code = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if not found_first_code:
            found_first_code = True
            continue # skip the imports cell
        if code_idx < len(code_cells_content):
            cell['source'] = code_cells_content[code_idx]
            code_idx += 1

with open("code/lab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
