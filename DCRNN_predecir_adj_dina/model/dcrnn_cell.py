from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from lib import utils


class DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(DCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        # supports_static: list of 2D sparse/dense supports used for all timesteps
        self._supports_static = []
        # supports_time: list of 3D tensors (seq_len, N, N) providing per-timestep supports
        self._supports_time = []
        self._use_gc_for_ru = use_gc_for_ru
        supports = []
        # If adj_mx is a Tensor (placeholder), build supports using TF ops so they
        # can be fed dynamically at runtime. Otherwise fall back to numpy/scipy path.
        if isinstance(adj_mx, tf.Tensor):
            adj = adj_mx
            # Distinguish between shapes:
            # 4D adj: (batch_size, T, N, N) -> per-sample per-timestep supports
            # 3D adj: (T, N, N) -> per-timestep supports
            # 2D adj: (N, N) -> static support
            nd = adj.get_shape().ndims
            if nd == 4:
                # per-sample, per-timestep adjacency: (B, T, N, N)
                if filter_type == "laplacian":
                    # compute per (batch* T)
                    B = tf.shape(adj)[0]
                    T = tf.shape(adj)[1]
                    adj_resh = tf.reshape(adj, (-1, self._num_nodes, self._num_nodes))
                    d = tf.reduce_sum(adj_resh, axis=2)
                    d_inv_sqrt = tf.pow(d, -0.5)
                    d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)
                    D_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
                    I = tf.tile(tf.expand_dims(tf.eye(self._num_nodes), 0), [tf.shape(adj_resh)[0], 1, 1])
                    normalized_laplacian = I - tf.matmul(tf.matmul(D_inv_sqrt, adj_resh), D_inv_sqrt)
                    normalized_laplacian = tf.reshape(normalized_laplacian, (B, T, self._num_nodes, self._num_nodes))
                    self._supports_time.append(normalized_laplacian)
                elif filter_type == "random_walk" or filter_type == "dual_random_walk":
                    B = tf.shape(adj)[0]
                    T = tf.shape(adj)[1]
                    adj_resh = tf.reshape(adj, (-1, self._num_nodes, self._num_nodes))
                    d = tf.reduce_sum(adj_resh, axis=2)
                    d_inv = tf.pow(d, -1.0)
                    d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)
                    D_inv = tf.linalg.diag(d_inv)
                    rw = tf.matmul(D_inv, adj_resh)
                    rw = tf.reshape(rw, (B, T, self._num_nodes, self._num_nodes))
                    # store as (B, T, N, N)
                    self._supports_time.append(rw)
                    if filter_type == "dual_random_walk":
                        adj_t = tf.transpose(adj, perm=[0, 1, 3, 2])
                        adj_t_resh = tf.reshape(adj_t, (-1, self._num_nodes, self._num_nodes))
                        d_rev = tf.reduce_sum(adj_t_resh, axis=2)
                        d_rev_inv = tf.pow(d_rev, -1.0)
                        d_rev_inv = tf.where(tf.math.is_inf(d_rev_inv), tf.zeros_like(d_rev_inv), d_rev_inv)
                        D_rev_inv = tf.linalg.diag(d_rev_inv)
                        rw_rev = tf.matmul(D_rev_inv, adj_t_resh)
                        rw_rev = tf.reshape(rw_rev, (B, T, self._num_nodes, self._num_nodes))
                        self._supports_time.append(rw_rev)
                else:
                    # fallback: store raw
                    self._supports_time.append(adj)
            elif nd == 3:
                # per-timestep adjacency: adj shape (T, N, N)
                if filter_type == "laplacian":
                    d = tf.reduce_sum(adj, axis=2)  # (T, N)
                    d_inv_sqrt = tf.pow(d, -0.5)
                    d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)
                    D_inv_sqrt = tf.linalg.diag(d_inv_sqrt)  # (T, N, N)
                    I = tf.tile(tf.expand_dims(tf.eye(self._num_nodes), 0), [tf.shape(adj)[0], 1, 1])
                    normalized_laplacian = I - tf.matmul(tf.matmul(D_inv_sqrt, adj), D_inv_sqrt)
                    self._supports_time.append(normalized_laplacian)
                elif filter_type == "random_walk":
                    d = tf.reduce_sum(adj, axis=2)  # (T, N)
                    d_inv = tf.pow(d, -1.0)
                    d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)
                    D_inv = tf.linalg.diag(d_inv)
                    random_walk_mx = tf.matmul(D_inv, adj)
                    # transpose per timestep
                    self._supports_time.append(tf.transpose(random_walk_mx, perm=[0, 2, 1]))
                elif filter_type == "dual_random_walk":
                    d = tf.reduce_sum(adj, axis=2)
                    d_inv = tf.pow(d, -1.0)
                    d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)
                    D_inv = tf.linalg.diag(d_inv)
                    rw = tf.matmul(D_inv, adj)
                    self._supports_time.append(tf.transpose(rw, perm=[0, 2, 1]))
                    # reverse random walk
                    adj_t = tf.transpose(adj, perm=[0, 2, 1])
                    d_rev = tf.reduce_sum(adj_t, axis=2)
                    d_rev_inv = tf.pow(d_rev, -1.0)
                    d_rev_inv = tf.where(tf.math.is_inf(d_rev_inv), tf.zeros_like(d_rev_inv), d_rev_inv)
                    D_rev_inv = tf.linalg.diag(d_rev_inv)
                    rw_rev = tf.matmul(D_rev_inv, adj_t)
                    self._supports_time.append(tf.transpose(rw_rev, perm=[0, 2, 1]))
                else:
                    self._supports_time.append(adj)
            else:
                # adj is 2D: use previous dense single-support logic
                if filter_type == "laplacian":
                    d = tf.reduce_sum(adj, axis=1)
                    d_inv_sqrt = tf.pow(d, -0.5)
                    d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)
                    D_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
                    normalized_laplacian = tf.eye(self._num_nodes) - tf.matmul(tf.matmul(D_inv_sqrt, adj), D_inv_sqrt)
                    self._supports_static.append(normalized_laplacian)
                elif filter_type == "random_walk":
                    d = tf.reduce_sum(adj, axis=1)
                    d_inv = tf.pow(d, -1.0)
                    d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)
                    D_inv = tf.linalg.diag(d_inv)
                    random_walk_mx = tf.matmul(D_inv, adj)
                    self._supports_static.append(tf.transpose(random_walk_mx))
                elif filter_type == "dual_random_walk":
                    d = tf.reduce_sum(adj, axis=1)
                    d_inv = tf.pow(d, -1.0)
                    d_inv = tf.where(tf.math.is_inf(d_inv), tf.zeros_like(d_inv), d_inv)
                    D_inv = tf.linalg.diag(d_inv)
                    rw = tf.matmul(D_inv, adj)
                    self._supports_static.append(tf.transpose(rw))
                    self._supports_static.append(tf.transpose(tf.matmul(D_inv, tf.transpose(adj))))
                else:
                    self._supports_static.append(adj)
        else:
            if filter_type == "laplacian":
                supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
            elif filter_type == "random_walk":
                supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            elif filter_type == "dual_random_walk":
                supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
                supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
            else:
                supports.append(utils.calculate_scaled_laplacian(adj_mx))
            for support in supports:
                self._supports_static.append(self._build_sparse_matrix(support))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "dcgru_cell"):
            # If inputs is a tuple (features, time_index), unpack time_index so
            # graph convolution can select the per-timestep support. time is a
            # vector of shape (batch_size,) containing the same timestep index.
            time = None
            if isinstance(inputs, (list, tuple)):
                features, time = inputs
            else:
                features = inputs

            with tf.variable_scope("gates"):  # Reset gate and update gate.
                output_size = 2 * self._num_units
                # We start with bias of 1.0 to not reset and not update.
                if self._use_gc_for_ru:
                    fn = self._gconv
                else:
                    fn = self._fc
                value = tf.nn.sigmoid(fn(features, state, output_size, bias_start=1.0, time=time))
                value = tf.reshape(value, (-1, self._num_nodes, output_size))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_nodes * self._num_units))
                u = tf.reshape(u, (-1, self._num_nodes * self._num_units))
            with tf.variable_scope("candidate"):
                c = self._gconv(features, r * state, self._num_units, time=time)
                if self._activation is not None:
                    c = self._activation(c)
            output = new_state = u * state + (1 - u) * c
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0, time=None):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        # Work with batch-first tensors: (batch_size, num_nodes, input_size)
        x0_batch = x  # shape: (batch_size, num_nodes, input_size)
        batch_size_tensor = tf.shape(x0_batch)[0]

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                # Determine time index if per-t supports need to be gathered.
                time_idx = None
                try:
                    time_idx = locals().get('time', None)
                except Exception:
                    time_idx = None

                # We'll collect tensors of shape (batch_size, num_nodes, input_size) for each diffusion matrix.
                mats = [x0_batch]

                # First handle time-varying supports (gather for the current timestep)
                x0_orig = x0_batch
                for support_time in self._supports_time:
                    st_nd = support_time.get_shape().ndims
                    if time_idx is not None:
                        t_index = tf.cast(time_idx[0], tf.int32)
                    else:
                        t_index = None

                    # support_time can be (T, N, N) or (B, T, N, N)
                    if st_nd == 4:
                        # (B, T, N, N) -> select per-sample support for timestep
                        if t_index is not None:
                            support_t = support_time[:, t_index, :, :]  # (B, N, N)
                        else:
                            support_t = support_time[:, tf.shape(support_time)[1] - 1, :, :]
                        # Batched matmul: support_t @ x0_batch -> (B, N, input_size)
                        x1 = tf.matmul(support_t, x0_orig)
                        mats.append(x1)
                        x0_local = x0_orig
                        for k in range(2, self._max_diffusion_step + 1):
                            x2 = 2 * tf.matmul(support_t, x1) - x0_local
                            mats.append(x2)
                            x1, x0_local = x2, x1
                    else:
                        # st_nd == 3: (T, N, N) -> select single (N,N) and tile per batch
                        if t_index is not None:
                            support_t_2d = tf.gather(support_time, t_index)  # (N, N)
                        else:
                            support_t_2d = support_time[tf.shape(support_time)[0] - 1]
                        support_t = tf.tile(tf.expand_dims(support_t_2d, 0), [batch_size_tensor, 1, 1])
                        x1 = tf.matmul(support_t, x0_orig)
                        mats.append(x1)
                        x0_local = x0_orig
                        for k in range(2, self._max_diffusion_step + 1):
                            x2 = 2 * tf.matmul(support_t, x1) - x0_local
                            mats.append(x2)
                            x1, x0_local = x2, x1

                # Then handle static supports (shared across all timesteps)
                for support in self._supports_static:
                    if isinstance(support, tf.SparseTensor):
                        support_dense = tf.sparse_tensor_to_dense(support)
                        support_b = tf.tile(tf.expand_dims(support_dense, 0), [batch_size_tensor, 1, 1])
                        x1 = tf.matmul(support_b, x0_orig)
                    else:
                        # support can be (N,N)
                        s_nd = support.get_shape().ndims
                        if s_nd == 2:
                            support_b = tf.tile(tf.expand_dims(support, 0), [batch_size_tensor, 1, 1])
                        else:
                            support_b = support
                        x1 = tf.matmul(support_b, x0_orig)
                    mats.append(x1)
                    x0_local = x0_orig
                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.matmul(support_b, x1) - x0_local
                        mats.append(x2)
                        x1, x0_local = x2, x1

            # concatenate mats -> shape (batch_size, num_nodes, input_size * num_matrices)
            num_matrices = len(mats)
            x = tf.concat(mats, axis=2)
            x = tf.reshape(x, shape=[batch_size_tensor * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
