#!/usr/bin/env python3
"""Nested time-series cross-validation for DCRNN student-position dataset.

Creates an outer holdout (train_ratio, default 0.7) and performs inner
TimeSeriesSplit on the training portion to select hyperparameters. For each
inner train/val split this script writes temporary `train.npz`/`val.npz`/`test.npz`
and calls the existing `DCRNNSupervisor` using the provided YAML config
overrides.

Usage example:
  python scripts/nested_cv.py --config data/student_nodes/config/dcrnn_students_finetune2.yaml \
      --outer-folds 1 --inner-splits 2 --epochs 1 --use_cpu_only True
"""
import argparse
import os
import shutil
import tempfile
import time
import json

import numpy as np
import yaml
import tensorflow as tf
import sys
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

from model.dcrnn_supervisor import DCRNNSupervisor


def load_concat_data(dataset_dir):
    # load train/val/test npz and concatenate in time order
    parts = []
    for name in ['train.npz', 'val.npz', 'test.npz']:
        p = os.path.join(dataset_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        d = np.load(p)
        parts.append((d['x'], d['y']))
    x_all = np.concatenate([p[0] for p in parts], axis=0)
    y_all = np.concatenate([p[1] for p in parts], axis=0)
    return x_all, y_all


def save_npz_triplet(outdir, x_train, y_train, x_val, y_val, x_test, y_test):
    os.makedirs(outdir, exist_ok=True)
    # Use the same filenames expected by `lib.utils.load_dataset` (train.npz, val.npz, test.npz)
    np.savez_compressed(os.path.join(outdir, 'train.npz'), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(outdir, 'val.npz'), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(outdir, 'test.npz'), x=x_test, y=y_test)


def build_param_grid(grid_file=None):
    if grid_file:
        with open(grid_file, 'r') as f:
            grid = json.load(f)
        return list(ParameterGrid(grid))
    # default small grid
    grid = {
        'rnn_units': [32, 64],
        'dropout': [0.0, 0.1],
        'max_diffusion_step': [1, 2],
        'filter_type': ['random_walk']
    }
    return list(ParameterGrid(grid))


def run_nested_cv(base_config, dataset_dir, out_root, outer_folds=1, inner_splits=3, train_ratio=0.7,
                  param_grid=None, epochs_override=None, use_cpu_only=True):
    x_all, y_all = load_concat_data(dataset_dir)
    N = x_all.shape[0]
    split = int(N * train_ratio)
    train_idx = np.arange(0, split)
    test_idx = np.arange(split, N)

    if param_grid is None:
        param_grid = build_param_grid(None)

    os.makedirs(out_root, exist_ok=True)
    results = []

    # ==== NUEVO: contadores de progreso ====
    total_hp = len(param_grid)
    total_inner_runs = outer_folds * inner_splits * total_hp
    print('Total hyperparameter configs:', total_hp)
    print('Total inner trainings to run:', total_inner_runs)
    run_id = 0
    # =======================================

    # We'll support outer_folds >1 by repeating the same holdout (user requested 70/30)
    for outer_i in range(outer_folds):
        print('Outer fold', outer_i)
        X_train_outer, Y_train_outer = x_all[train_idx], y_all[train_idx]
        X_test_outer, Y_test_outer = x_all[test_idx], y_all[test_idx]

        # Inner CV on X_train_outer
        tscv = TimeSeriesSplit(n_splits=inner_splits)
        best_hp = None
        best_score = float('inf')

        for hp_idx, hp in enumerate(param_grid, start=1):
            inner_scores = []
            for split_idx, (inner_train_rel, inner_val_rel) in enumerate(tscv.split(X_train_outer), start=1):
                # ==== NUEVO: imprimir progreso por run ====
                run_id += 1
                print(
                    f"[Progress] Run {run_id}/{total_inner_runs} | "
                    f"Outer {outer_i + 1}/{outer_folds} | "
                    f"HP {hp_idx}/{total_hp} | "
                    f"Split {split_idx}/{inner_splits}",
                    flush=True
                )
                # ==========================================

                # inner indices relative to X_train_outer
                X_tr = X_train_outer[inner_train_rel]
                Y_tr = Y_train_outer[inner_train_rel]
                X_va = X_train_outer[inner_val_rel]
                Y_va = Y_train_outer[inner_val_rel]

                tmpdir = tempfile.mkdtemp(prefix='nestedcv_')
                try:
                    save_npz_triplet(tmpdir, X_tr, Y_tr, X_va, Y_va, X_va, Y_va)

                    cfg = dict(base_config)
                    # ensure nested dicts are copied
                    cfg = yaml.safe_load(yaml.dump(cfg))
                    cfg['data']['dataset_dir'] = tmpdir
                    # avoid restoring from existing checkpoints when running nested CV
                    if 'train' not in cfg:
                        cfg['train'] = {}
                    cfg['train']['model_filename'] = None
                    # set unique log_dir per inner split to avoid collisions
                    cfg['train']['log_dir'] = os.path.join(
                        cfg.get('log_dir', './logs'),
                        'nestedcv_inner_%d' % int(time.time())
                    )
                    # override model params from hp
                    for k, v in hp.items():
                        cfg['model'][k] = v
                    if epochs_override is not None:
                        cfg['train']['epochs'] = int(epochs_override)
                    # force CPU if requested
                    cfg['use_cpu_only'] = bool(use_cpu_only)

                    tf.reset_default_graph()
                    sess = tf.Session()
                    sup = DCRNNSupervisor(adj_mx=None, **cfg)
                    sup.train(sess)
                    # Evaluate on val loader
                    val_results = sup.run_epoch_generator(
                        sess,
                        sup._test_model,
                        sup._data['val_loader'].get_iterator(),
                        training=False
                    )
                    inner_scores.append(val_results['loss'])
                    sess.close()
                finally:
                    shutil.rmtree(tmpdir)
            avg = float(np.mean(inner_scores))
            print('  hp', hp, 'avg val loss', avg)
            if avg < best_score:
                best_score = avg
                best_hp = hp

        print('Best hp for outer fold', outer_i, best_hp, 'score', best_score)

        # Retrain on full outer train with best_hp and evaluate on outer test
        tmpdir = tempfile.mkdtemp(prefix='nestedcv_final_')
        try:
            save_npz_triplet(tmpdir, X_train_outer, Y_train_outer,
                             X_test_outer, Y_test_outer,
                             X_test_outer, Y_test_outer)
            cfg = dict(base_config)
            cfg = yaml.safe_load(yaml.dump(cfg))
            cfg['data']['dataset_dir'] = tmpdir
            if 'train' not in cfg:
                cfg['train'] = {}
            cfg['train']['model_filename'] = None
            cfg['train']['log_dir'] = os.path.join(
                cfg.get('log_dir', './logs'),
                'nestedcv_outer_%d' % outer_i
            )
            for k, v in best_hp.items():
                cfg['model'][k] = v
            if epochs_override is not None:
                cfg['train']['epochs'] = int(epochs_override)
            cfg['use_cpu_only'] = bool(use_cpu_only)

            tf.reset_default_graph()
            sess = tf.Session()
            sup = DCRNNSupervisor(adj_mx=None, **cfg)
            sup.train(sess)
            # Evaluate on test loader and collect predictions
            test_outputs = sup.evaluate(sess)
            # save model config and results
            fold_dir = os.path.join(out_root, 'outer_%d' % outer_i)
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, 'best_hp.json'), 'w') as f:
                json.dump(best_hp, f)
            with open(os.path.join(fold_dir, 'results.json'), 'w') as f:
                json.dump({'best_val_loss': best_score}, f)
            results.append({'outer': outer_i, 'best_hp': best_hp, 'best_val_loss': best_score})
            sess.close()
        finally:
            shutil.rmtree(tmpdir)

    # summary
    out_file = os.path.join(out_root, 'nested_cv_summary.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print('Nested CV finished. Summary saved to', out_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Base YAML config file')
    parser.add_argument('--dataset_dir', type=str, default='data/student_nodes', help='Directory with train/val/test npz')
    parser.add_argument('--out_root', type=str, default='logs/nested_cv', help='Output directory for nested CV results')
    parser.add_argument('--outer-folds', type=int, default=1)
    parser.add_argument('--inner-splits', type=int, default=3)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--grid-file', type=str, default=None, help='JSON file with param grid for ParameterGrid')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs in config for quick runs')
    parser.add_argument('--use_cpu_only', action='store_true')
    args = parser.parse_args()

    base_config = yaml.safe_load(open(args.config))
    param_grid = build_param_grid(args.grid_file)
    run_nested_cv(base_config, args.dataset_dir, args.out_root,
                  outer_folds=args.outer_folds, inner_splits=args.inner_splits,
                  train_ratio=args.train_ratio, param_grid=param_grid,
                  epochs_override=args.epochs, use_cpu_only=args.use_cpu_only)


if __name__ == '__main__':
    main()
