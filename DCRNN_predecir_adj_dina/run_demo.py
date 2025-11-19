import argparse
import numpy as np
import os
import sys
import yaml

# TensorFlow import shim: prefer TF2 GPU builds if present but run the code
# in TF1 compatibility mode using `tf.compat.v1`. This allows installing a
# TF2.x GPU wheel compatible with your system CUDA while keeping TF1 code.
import tensorflow as _tf
if _tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    tf = _tf

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    graph_pkl_filename = config['data'].get('graph_pkl_filename')
    # If dynamic adjacency is requested, do not load a static graph; the supervisor
    # will create an adjacency placeholder to be fed per-batch.
    if config['data'].get('dynamic_adj', False):
        adj_mx = None
    else:
        _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        model_filename = config['train'].get('model_filename')
        if model_filename is not None:
            supervisor.load(sess, model_filename)
        outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.output_filename, **outputs)
        print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
