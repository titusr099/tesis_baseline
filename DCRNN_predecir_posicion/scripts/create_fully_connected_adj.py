"""Create a fully-connected adjacency pickle for student nodes.

Pickle format: (sensor_ids, sensor_id_to_ind, adj_mx)
"""
import pickle
import numpy as np
import argparse
import os


def make_fully_connected(num_nodes):
    sensor_ids = [f"S{i:02d}" for i in range(num_nodes)]
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    adj = np.ones((num_nodes, num_nodes), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    return sensor_ids, sensor_id_to_ind, adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=10)
    parser.add_argument('--output_pkl', type=str, default='data/student_nodes/adj_mx.pkl')
    args = parser.parse_args()
    sensor_ids, sensor_id_to_ind, adj = make_fully_connected(args.num_nodes)
    outdir = os.path.dirname(args.output_pkl)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output_pkl, 'wb') as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj), f)
    print('Wrote', args.output_pkl)


if __name__ == '__main__':
    main()
