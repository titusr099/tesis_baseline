"""Generate train/val/test .npz for student-position prediction.

This script synthesizes 2D trajectories for N students,
builds input features per node [x, y, time_of_day], and saves compressed npz files with
sliding windows ready for DCRNN training.

Output files (in output_dir): train.npz, val.npz, test.npz with keys: x, y, x_offsets, y_offsets
x shape: (N_samples, Tx, num_nodes, input_dim)
y shape: (N_samples, Ty, num_nodes, output_dim)
"""
import os
import numpy as np
import pandas as pd
import argparse


def _get_rng(seed=None, existing=None):
    """Return a random number generator compatible with old/new numpy.

    If `existing` is provided, return it. Otherwise try to use
    numpy.random.default_rng (new API) and fall back to
    numpy.random.RandomState for older numpy versions.
    """
    if existing is not None:
        return existing
    try:
        return np.random.default_rng(seed)
    except AttributeError:
        return np.random.RandomState(seed)


def smooth_noise(T, scale=1.0, alpha=0.05, rng=None):
    rng = _get_rng(None, existing=rng)
    e = rng.normal(0, scale, T)
    y = np.zeros(T, dtype=np.float32)
    for i in range(1, T):
        y[i] = alpha * e[i] + (1 - alpha) * y[i - 1]
    return y


def generate_students_positions(num_students=10, step_seconds=1, days=1, seed=42):
    STEPS_PER_DAY = (24 * 60 * 60) // step_seconds
    T = STEPS_PER_DAY * days
    start_time = pd.Timestamp("2025-01-01 08:00:00")
    idx = pd.date_range(start_time, periods=T, freq=f"{step_seconds}S")
    rng = _get_rng(seed)
    t = np.arange(T)
    positions_x = {}
    positions_y = {}
    period = STEPS_PER_DAY
    omega = 2 * np.pi / period
    for i in range(num_students):
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        amp_x = rng.uniform(1.0, 3.0)
        amp_y = rng.uniform(1.0, 3.0)
        drift_x = rng.uniform(-0.020, 0.002)
        drift_y = rng.uniform(-0.002, 0.002)
        base_x = amp_x * np.sin(omega * t + phase_x)
        base_y = amp_y * np.cos(omega * t + phase_y)
        drift = np.stack([drift_x * t, drift_y * t], axis=1)
        noise_x = smooth_noise(T, scale=0.3, alpha=0.05, rng=rng)
        noise_y = smooth_noise(T, scale=0.3, alpha=0.05, rng=rng)
        bias_x = rng.normal(0, 5)
        bias_y = rng.normal(0, 5)
        px = (base_x + drift[:, 0] + noise_x + bias_x).astype(np.float32)
        py = (base_y + drift[:, 1] + noise_y + bias_y).astype(np.float32)
        sid = f"S{i:02d}"
        positions_x[sid] = px
        positions_y[sid] = py
    X = pd.DataFrame(positions_x, index=idx)
    Y = pd.DataFrame(positions_y, index=idx)
    return X, Y


def generate_graph_seq2seq_io_from_positions(df_x, df_y, x_offsets, y_offsets, add_time_in_day=True):
    # df_x, df_y: DataFrames (T, N)
    num_samples, num_nodes = df_x.shape
    data_pos = np.stack([df_x.values, df_y.values], axis=-1)  # (T, N, 2)
    data_list = [data_pos]
    if add_time_in_day:
        time_ind = (df_x.index.values - df_x.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day.astype(np.float32))
    data = np.concatenate(data_list, axis=-1)  # (T, N, F)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def split_and_save(x, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_samples = x.shape[0]
    num_test = int(round(num_samples * 0.2))
    num_train = int(round(num_samples * 0.7))
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    # offsets saved for reproducibility
    # x_offsets and y_offsets are embedded in filenames by caller
    np.savez_compressed(os.path.join(output_dir, 'train.npz'), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(output_dir, 'val.npz'), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(output_dir, 'test.npz'), x=x_test, y=y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/student_nodes', help='Output directory')
    parser.add_argument('--num_students', type=int, default=10)
    parser.add_argument('--step_seconds', type=int, default=1)
    parser.add_argument('--days', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    args = parser.parse_args()

    X, Y = generate_students_positions(num_students=args.num_students, step_seconds=args.step_seconds,
                                       days=args.days, seed=args.seed)
    x_offsets = np.arange(-args.seq_len + 1, 1, 1)
    y_offsets = np.arange(1, args.horizon + 1, 1)
    x, y = generate_graph_seq2seq_io_from_positions(X, Y, x_offsets, y_offsets, add_time_in_day=True)
    # Save with offsets
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, 'train.npz'), x=x[:int(len(x)*0.7)], y=y[:int(len(y)*0.7)], x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]), y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))
    np.savez_compressed(os.path.join(out, 'val.npz'), x=x[int(len(x)*0.7):int(len(x)*0.8)], y=y[int(len(y)*0.7):int(len(y)*0.8)], x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]), y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))
    np.savez_compressed(os.path.join(out, 'test.npz'), x=x[-int(len(x)*0.2):], y=y[-int(len(y)*0.2):], x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]), y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))
    print('Saved train/val/test to', out)


if __name__ == '__main__':
    main()
