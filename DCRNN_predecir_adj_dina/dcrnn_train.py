from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import yaml

# TensorFlow import shim: prefer TF2 GPU builds if present but run the code
# in TF1 compatibility mode using `tf.compat.v1`. This lets you install a
# modern `tensorflow` wheel (TF2.x GPU) compatible with your CUDA version
# without changing the TF1-style code in this repo.
import tensorflow as _tf
if _tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    tf = _tf

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        # If dynamic adjacency is requested, do not load a static graph here.
        if supervisor_config['data'].get('dynamic_adj', False):
            adj_mx = None
        else:
            sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

            supervisor.train(sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
