import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

npz_path = Path('/home/rlara/trabajo_rl/DCRNN_predecir_adj_dina/data/dcrnn_predictions.npz')
if not npz_path.exists():
    raise SystemExit('Predictions file not found: {}'.format(npz_path))

npz = np.load(str(npz_path), allow_pickle=True)
preds = npz['predictions']  # shape (H, Nsamples, Nnodes, output_dim)
truths = npz['groundtruth']

print('preds shape', preds.shape)
H, Nsamples, Nnodes, C = preds.shape

sample_idx = 0
node_idx = 0
outdir = Path('logs/dcrnn_students_expt1/plots')
outdir.mkdir(parents=True, exist_ok=True)

# For each channel (e.g., x,y) plot pred vs true across horizons
horizons = np.arange(1, H+1)
for c in range(C):
    pred_vals = np.array([preds[h, sample_idx, node_idx, c] for h in range(H)])
    true_vals = np.array([truths[h, sample_idx, node_idx, c] for h in range(H)])

    plt.figure(figsize=(8,4))
    plt.plot(horizons, true_vals, '-o', label='truth')
    plt.plot(horizons, pred_vals, '-s', label='pred')
    plt.xlabel('Horizon')
    plt.ylabel('Value (channel {})'.format(c))
    plt.title(f'Sample {sample_idx}, Node {node_idx}, Channel {c}: pred vs truth by horizon')
    plt.grid(True)
    plt.legend()
    fname = outdir / f'sample{sample_idx}_node{node_idx}_ch{c}.png'
    plt.savefig(str(fname), bbox_inches='tight')
    plt.close()
    print('Saved', fname)

# Also save scatter of pred vs truth across horizons (all channels concatenated)
plt.figure(figsize=(5,5))
all_pred = preds[:, sample_idx, node_idx, :].reshape(-1)
all_true = truths[:, sample_idx, node_idx, :].reshape(-1)
plt.scatter(all_true, all_pred, alpha=0.6)
mn = min(all_true.min(), all_pred.min())
mx = max(all_true.max(), all_pred.max())
plt.plot([mn,mx], [mn,mx], 'r--')
plt.xlabel('truth')
plt.ylabel('pred')
plt.title(f'Sample {sample_idx}, Node {node_idx}: pred vs truth scatter')
fname2 = outdir / f'sample{sample_idx}_node{node_idx}_scatter.png'
plt.savefig(str(fname2), bbox_inches='tight')
plt.close()
print('Saved', fname2)
