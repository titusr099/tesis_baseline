#!/usr/bin/env python3
import numpy as np
import os
from pathlib import Path

base = Path(__file__).resolve().parent
files = sorted([p for p in base.glob('*.npz')])

print('Base dir:', base)

def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = labels != null_val
        mask = mask.astype('float32')
        # avoid division by zero if all masked
        mean_mask = mask.mean() if mask.mean() != 0 else 1.0
        mask /= mean_mask
        mae = np.abs(preds - labels).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return float(np.mean(mae))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = labels != null_val
        mask = mask.astype('float32')
        mean_mask = mask.mean() if mask.mean() != 0 else 1.0
        mask /= mean_mask
        mse = np.square(preds - labels).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return float(np.mean(mse))


def masked_rmse_np(preds, labels, null_val=np.nan):
    return float(np.sqrt(masked_mse_np(preds, labels, null_val=null_val)))


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = labels != null_val
        mask = mask.astype('float32')
        mean_mask = mask.mean() if mask.mean() != 0 else 1.0
        mask /= mean_mask
        # safe division
        denom = labels.astype('float32')
        mape = np.abs((preds - labels).astype('float32') / denom)
        mape = np.nan_to_num(mask * mape)
        return float(np.mean(mape))


npz_data = {}
for f in files:
    print('\n---', f.name)
    try:
        data = np.load(f, allow_pickle=True)
    except Exception as e:
        print('  ERROR loading', e)
        continue
    keys = getattr(data, 'files', None)
    if keys:
        print(' keys:', keys)
    else:
        print(' single array or unknown keys')
    for k in keys:
        arr = data[k]
        print(f'  - {k}: shape={arr.shape}, dtype={arr.dtype}')
        # basic stats
        try:
            flat = arr.ravel()
            finite = flat[np.isfinite(flat)]
            print(f'     count={flat.size}, nans={np.isnan(flat).sum()}, infs={np.isinf(flat).sum()}')
            if finite.size>0 and np.issubdtype(finite.dtype, np.number):
                print(f'     min={finite.min():.6g}, max={finite.max():.6g}, mean={finite.mean():.6g}, std={finite.std():.6g}')
        except Exception as e:
            print('    stats error', e)
    npz_data[f.name] = data

# Try to find prediction vs truth
# Common names: 'predictions', 'y_hat', 'pred', 'targets', 'labels', or arr_0
pred_name = None
truth_name = None
for name, data in npz_data.items():
    files_list = getattr(data, 'files', [])
    # heuristics
    if 'dcrnn_predictions.npz' in name:
        pred_name = name
    if 'dists_truth' in name or name.startswith('dists_truth'):
        truth_name = name
    if 'test.npz' in name:
        # test.npz may contain 'x' or 'y' or 'test'
        if truth_name is None:
            truth_name = name

print('\nHeuristics found: pred_name=', pred_name, ' truth_name=', truth_name)

# If we have predictions and truth arrays with compatible shapes, compute metrics
if pred_name and truth_name:
    preds_npz = npz_data[pred_name]
    truth_npz = npz_data[truth_name]
    # try various key matching
    found = False
    for pk in getattr(preds_npz, 'files', []):
        for tk in getattr(truth_npz, 'files', []):
            p = preds_npz[pk]
            t = truth_npz[tk]
            if p.shape == t.shape:
                print(f'\nComparing pred {pred_name}:{pk} to truth {truth_name}:{tk} shape={p.shape}')
                try:
                    mae = masked_mae_np(p, t)
                    rmse = masked_rmse_np(p, t)
                    mape = masked_mape_np(p, t)
                except Exception as e:
                    print('  error computing metrics:', e)
                    continue
                print(f'  MAE={mae:.6g}, RMSE={rmse:.6g}, MAPE={mape:.6g}')
                found = True
    if not found:
        print('\nNo matching key pairs with equal shapes found between pred and truth files.')
else:
    print('\nNo pred/truth pair detected automatically. Si quieres, indícame qué archivo comparar.')

# If there is pct_err.npz, print its content
if 'pct_err.npz' in npz_data:
    p = npz_data['pct_err.npz']
    for k in getattr(p, 'files', []):
        arr = p[k]
        print('\nPct err file key', k, 'shape', arr.shape)
        flat = arr.ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size>0:
            print(f'  pct err min={finite.min():.6g}, max={finite.max():.6g}, mean={finite.mean():.6g}, std={finite.std():.6g}')

print('\nDone.')
