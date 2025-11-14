#!/usr/bin/env python3
"""Genera figuras para la presentación: grafo inicial, evolución temporal, predicción vs verdad,
matrices de distancia y métricas.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import generator by path (scripts/ is not a package)
import importlib.util
base_repo = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('generate_students_training_data',
                                              str(base_repo / 'scripts' / 'generate_students_training_data.py'))
gmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gmod)
generate_students_positions = gmod.generate_students_positions


OUT_DIR = Path(__file__).resolve().parent / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_initial_graph(positions, title='Grafo inicial (t=0)', fname='graph_initial.png'):
    # positions: (V, 2)
    V = positions.shape[0]
    dists = np.linalg.norm(positions[None, :, :] - positions[:, None, :], axis=-1)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(positions[:,0], positions[:,1], s=80, c='C0')
    for i in range(V):
        ax.text(positions[i,0], positions[i,1], f'{i}', fontsize=9, va='bottom')
    # draw edges with alpha ~ 1/(1+dist)
    for i in range(V):
        for j in range(i+1, V):
            x = [positions[i,0], positions[j,0]]
            y = [positions[i,1], positions[j,1]]
            alpha = 0.1 + 0.9 * (1.0 - min(1.0, dists[i,j] / (dists.mean()*2)))
            ax.plot(x, y, c='gray', alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.axis('equal')
    path = OUT_DIR / fname
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('Saved', path)


def plot_evolution(positions_all, indices=None, fname_prefix='evo'):
    # positions_all: (T, V, 2)
    T, V, _ = positions_all.shape
    if indices is None:
        # pick up to 6 evenly spaced snapshots
        indices = np.linspace(0, min(60, T-1), min(6, T)).astype(int)
    for idx in indices:
        pos = positions_all[idx]
        plot_initial_graph(pos, title=f'Grafo t={idx}', fname=f'{fname_prefix}_{idx:03d}.png')


def plot_pred_vs_truth(test_npz, pred_npz, sample_index=0, nodes=(0,1,2)):
    # Load test y: (N, H, V, 3) ; preds: (H, N, V, 2)
    y = test_npz['y']  # shape (N, H, V, 3)
    preds = pred_npz['predictions']  # (H, N, V, 2)
    # transpose preds -> (N, H, V, 2)
    preds_t = preds.transpose(1,0,2,3)
    N, H, V, _ = preds_t.shape
    # choose sample
    if sample_index >= N:
        sample_index = 0
    t = np.arange(H)
    for node in nodes:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(t, y[sample_index,:,:,0][:,node], '-o', label='truth x')
        ax.plot(t, y[sample_index,:,:,1][:,node], '-o', label='truth y')
        ax.plot(t, preds_t[sample_index,:,:,0][:,node], '--x', label='pred x')
        ax.plot(t, preds_t[sample_index,:,:,1][:,node], '--x', label='pred y')
        ax.set_title(f'Sample {sample_index} Node {node} (horizon)')
        ax.set_xlabel('horizon step'); ax.set_ylabel('position')
        ax.legend()
        path = OUT_DIR / f'pred_vs_truth_sample{sample_index}_node{node}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('Saved', path)


def plot_distance_matrices(dists_truth, dists_pred):
    # dists_*: (N, H, V, V)
    # compute mean across N to get average adjacency per horizon
    mean_truth = np.nanmean(dists_truth, axis=0)  # (H, V, V)
    mean_pred = np.nanmean(dists_pred, axis=0)
    H = mean_truth.shape[0]
    # plot heatmap for horizon 0 and difference
    import seaborn as sns
    sns.set()
    for h in [0, min(3, H-1), H-1]:
        fig, ax = plt.subplots(1,3, figsize=(15,4))
        sns.heatmap(mean_truth[h], ax=ax[0], cmap='viridis')
        ax[0].set_title(f'mean truth dist (h={h})')
        sns.heatmap(mean_pred[h], ax=ax[1], cmap='viridis')
        ax[1].set_title(f'mean pred dist (h={h})')
        sns.heatmap(mean_pred[h]-mean_truth[h], ax=ax[2], cmap='bwr', center=0)
        ax[2].set_title('pred - truth')
        path = OUT_DIR / f'dists_mean_h{h}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('Saved', path)


def main():
    # 1) regenerate synthetic positions (same generator as used para crear datos)
    X, Y = generate_students_positions(num_students=10, step_seconds=1, days=2, seed=42)
    # X, Y are DataFrames (T, N)
    # Simplify: use X and Y together
    pos_x = X.values  # (T, V)
    pos_y = Y.values
    positions_all = np.stack([pos_x, pos_y], axis=-1)  # (T, V, 2)
    # plot initial graph
    plot_initial_graph(positions_all[0], title='Grafo inicial (pos t=0)', fname='graph_initial_positions_t0.png')
    # plot several snapshots
    plot_evolution(positions_all, indices=np.linspace(0, min(200, positions_all.shape[0]-1), 6).astype(int), fname_prefix='evo')

    # 2) plot model predictions vs truth
    base = Path(__file__).resolve().parent
    test_npz = np.load(base / 'test.npz', allow_pickle=True)
    pred_npz = np.load(base / 'dcrnn_predictions.npz', allow_pickle=True)
    plot_pred_vs_truth(test_npz, pred_npz, sample_index=0, nodes=(0,1,2))

    # 3) plot distance matrices
    dists_pred = np.load(base / 'dists_pred.npz', allow_pickle=True)['dists_pred']
    dists_truth = np.load(base / 'dists_truth.npz', allow_pickle=True)['dists_truth']
    plot_distance_matrices(dists_truth, dists_pred)

    # 4) print brief metrics summary
    preds = pred_npz['predictions']
    gts = pred_npz['groundtruth']
    # compute MAE/RMSE with standard masking
    mask = ~np.isnan(gts)
    mask = mask.astype('float32')
    mask /= mask.mean() if mask.mean()!=0 else 1.0
    mae = np.nan_to_num(np.abs(preds-gts) * mask).mean()
    rmse = np.sqrt(np.nan_to_num(((preds-gts)**2) * mask).mean())
    print('\nSUMMARY METRICS:')
    print('Predictions shape', preds.shape, 'Groundtruth shape', gts.shape)
    print(f'MAE={mae:.6g}, RMSE={rmse:.6g}')


if __name__ == '__main__':
    main()
