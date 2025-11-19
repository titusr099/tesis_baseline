from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import time
import yaml

from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mae_loss

from model.dcrnn_model import DCRNNModel


class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, adj_mx, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        # Data preparation
        self._data = utils.load_dataset(**self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        # If adj_mx is None we want to support dynamic adjacency via a placeholder.
        num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self._adj_ph = None
        self._per_t_adj = False
        if adj_mx is None:
            # Decide whether we want per-timestep dynamic adjacency
            batch_size = int(self._data_kwargs.get('batch_size'))
            if self._data_kwargs.get('per_t_dynamic_adj', False):
                seq_len = int(self._model_kwargs.get('seq_len'))
                # placeholder for sequence of adjacency matrices per sample: (batch_size, seq_len, N, N)
                self._adj_ph = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, num_nodes), name='adj_ph')
                self._per_t_adj = True
            else:
                # create a per-sample placeholder and pass it to both train/test models
                self._adj_ph = tf.placeholder(tf.float32, shape=(batch_size, num_nodes, num_nodes), name='adj_ph')
            adj_input = self._adj_ph
        else:
            adj_input = adj_mx

        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                self._train_model = DCRNNModel(is_training=True, scaler=scaler,
                                               batch_size=self._data_kwargs['batch_size'],
                                               adj_mx=adj_input, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._test_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['test_batch_size'],
                                              adj_mx=adj_input, **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr, )
        elif optimizer_name == 'amsgrad':
            optimizer = AMSGrad(self._lr, epsilon=epsilon)

        # Calculate loss
        output_dim = self._model_kwargs.get('output_dim')
        preds = self._train_model.outputs
        labels = self._train_model.labels[..., :output_dim]

        null_val = 0.
        self._loss_fn = masked_mae_loss(scaler, null_val)
        self._train_loss = self._loss_fn(preds=preds, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._epoch = 0
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, model, data_generator, return_output=False, training=False, writer=None):
        losses = []
        maes = []
        outputs = []
        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        fetches = {
            'loss': loss,
            'mae': loss,
            'global_step': tf.train.get_or_create_global_step()
        }
        if training:
            fetches.update({
                'train_op': self._train_op
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        # Prepare adjacency save directory
        adj_save_dir = self._data_kwargs.get('adj_save_dir', os.path.join(self._log_dir, 'adj_matrices'))
        try:
            os.makedirs(adj_save_dir)
        except Exception:
            pass

        batch_counter = 0

        for _, (x, y) in enumerate(data_generator):
            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            # If dynamic adjacency is enabled (we created a placeholder), compute
            # adjacency matrices from the batch positions and feed them.
            if hasattr(self, '_adj_ph') and self._adj_ph is not None:
                try:
                    seq_len = x.shape[1]
                    # For per-t adjacency compute each timestep adjacency per sample
                    per_t_adjs = []
                    for t in range(seq_len):
                        # pos: (batch_size, num_nodes, input_dim)
                        pos = x[:, t, :, :]
                        # Use per-sample positions (no averaging) -> coords: (batch_size, num_nodes, coord_dim)
                        if pos.shape[-1] >= 2:
                            coords = pos[..., :2]
                        else:
                            coords = pos[..., :1]
                        # compute pairwise diffs per sample: (batch_size, num_nodes, num_nodes, coord_dim)
                        diff = coords[:, :, None, :] - coords[:, None, :, :]
                        # distances per sample: (batch_size, num_nodes, num_nodes)
                        dists = np.linalg.norm(diff, axis=-1)
                        transform = self._data_kwargs.get('adjacency_transform', 'distance')
                        if transform == 'similarity':
                            sigma = float(self._data_kwargs.get('adjacency_sigma', 1.0))
                            adj_t = np.exp(-dists / sigma).astype(np.float32)
                        else:
                            adj_t = dists.astype(np.float32)
                        # adj_t is (batch_size, num_nodes, num_nodes)
                        per_t_adjs.append(adj_t)
                except Exception:
                    per_t_adjs = [np.eye(model._num_nodes, dtype=np.float32) for _ in range(x.shape[1])]

                if getattr(self, '_per_t_adj', False):
                    # Save a summary (mean over batch) per timestep and feed the per-sample stacked sequence
                    for t_idx, adj_t in enumerate(per_t_adjs):
                        try:
                            adj_mean = np.mean(adj_t, axis=0).astype(np.float32)
                            fname = os.path.join(adj_save_dir, 'batch_{:06d}_t_{:02d}.npz'.format(batch_counter, t_idx))
                            np.savez_compressed(fname, adj=adj_mean, batch=batch_counter, t=t_idx)
                        except Exception:
                            pass
                    try:
                        # Stack to shape (batch_size, seq_len, N, N)
                        adj_stack = np.stack(per_t_adjs, axis=1).astype(np.float32)
                    except Exception:
                        adj_stack = np.array(per_t_adjs, dtype=np.float32).transpose((1, 0, 2, 3))
                    feed_dict.update({self._adj_ph: adj_stack})
                else:
                    # choose which adjacency to feed for the whole batch (per-sample)
                    adj_feed_mode = self._data_kwargs.get('adjacency_feed_mode', 'last')
                    if adj_feed_mode == 'last':
                        adj_to_feed = per_t_adjs[-1]
                    elif adj_feed_mode == 'first':
                        adj_to_feed = per_t_adjs[0]
                    elif adj_feed_mode == 'mean_over_seq':
                        adj_to_feed = np.mean(np.stack(per_t_adjs, axis=1), axis=1).astype(np.float32)
                    else:
                        adj_to_feed = per_t_adjs[-1]
                    # adj_to_feed shape: (batch_size, N, N)
                    feed_dict.update({self._adj_ph: adj_to_feed})
                batch_counter += 1

            vals = sess.run(fetches, feed_dict=feed_dict)

            losses.append(vals['loss'])
            maes.append(vals['mae'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = train_kwargs.get('model_filename')
        if model_filename is not None:
            saver.restore(sess, model_filename)
            self._epoch = epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
        self._logger.info('Start training ...')

        while self._epoch <= epochs:
            # Learning rate schedule.
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True,
                                                     writer=self._writer)
            train_loss, train_mae = train_results['loss'], train_results['mae']
            if train_loss > 1e5:
                self._logger.warning('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self.run_epoch_generator(sess, self._test_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            val_loss, val_mae = np.asscalar(val_results['loss']), np.asscalar(val_results['mae'])

            utils.add_simple_summary(self._writer,
                                     ['loss/train_loss', 'metric/train_mae', 'loss/val_loss', 'metric/val_mae'],
                                     [train_loss, train_mae, val_loss, val_mae], global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} lr:{:.6f} {:.1f}s'.format(
                self._epoch, epochs, global_step, train_mae, val_mae, new_lr, (end_time - start_time))
            self._logger.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.evaluate(sess)
            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                self._logger.info(
                    'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def evaluate(self, sess, **kwargs):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, self._test_model,
                                                self._data['test_loader'].get_iterator(),
                                                return_output=True,
                                                training=False)

        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']
        utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)
        # Concatenate batch outputs -> shape (num_samples, horizon, num_nodes, output_dim)
        y_preds = np.concatenate(y_preds, axis=0)
        scaler = self._data['scaler']
        predictions = []
        y_truths = []
        # Determine output_dim
        output_dim = int(self._model_kwargs.get('output_dim', 1))

        # For each horizon, inverse-transform channel-wise and compute metrics.
        for horizon_i in range(self._data['y_test'].shape[1]):
            # raw scaled arrays: shape (N, num_nodes, input_dim)
            y_truth_raw = self._data['y_test'][:, horizon_i, :, :output_dim]
            y_pred_raw = y_preds[:y_truth_raw.shape[0], horizon_i, :, :output_dim]

            # Inverse transform per channel using scaler.mean / scaler.std
            # Resulting arrays: (N, num_nodes, output_dim)
            y_truth_unscaled = np.zeros_like(y_truth_raw)
            y_pred_unscaled = np.zeros_like(y_pred_raw)
            mean = getattr(scaler, 'mean')
            std = getattr(scaler, 'std')
            for k in range(output_dim):
                m = float(mean[k]) if hasattr(mean, '__len__') else float(mean)
                s = float(std[k]) if hasattr(std, '__len__') else float(std)
                y_truth_unscaled[:, :, k] = (y_truth_raw[:, :, k] * s) + m
                y_pred_unscaled[:, :, k] = (y_pred_raw[:, :, k] * s) + m

            # Flatten channel dimension if output_dim == 1 to keep compatibility with downstream helpers
            if output_dim == 1:
                y_truth_2d = y_truth_unscaled[:, :, 0]
                y_pred_2d = y_pred_unscaled[:, :, 0]
            else:
                # For multi-dim outputs, compute metrics averaged across channels
                y_truth_2d = y_truth_unscaled.reshape((y_truth_unscaled.shape[0], -1))
                y_pred_2d = y_pred_unscaled.reshape((y_pred_unscaled.shape[0], -1))

            # Compute metrics
            mae = metrics.masked_mae_np(y_pred_2d, y_truth_2d, null_val=0)
            mape = metrics.masked_mape_np(y_pred_2d, y_truth_2d, null_val=0)
            rmse = metrics.masked_rmse_np(y_pred_2d, y_truth_2d, null_val=0)
            self._logger.info(
                "Horizon {:02d}, MAE: {:.4f}, MAPE: {:.6f}, RMSE: {:.4f}".format(
                    horizon_i + 1, mae, mape, rmse
                )
            )
            utils.add_simple_summary(self._writer,
                                     ['%s_%d' % (item, horizon_i + 1) for item in
                                      ['metric/rmse', 'metric/mape', 'metric/mae']],
                                     [rmse, mape, mae],
                                     global_step=global_step)

            # Store per-horizon predictions and truths (unscaled)
            predictions.append(y_pred_unscaled)
            y_truths.append(y_truth_unscaled)

        outputs = {
            'predictions': predictions,
            'groundtruth': y_truths
        }
        return outputs

    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix, global_step=global_step,
                                                             write_meta_graph=False)
        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']
