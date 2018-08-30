#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import math
import functools
from op_test import OpTest
from test_lstm_op import ACTIVATION


def gru(
        input,  # T x 3D
        lod,  # 1 x N
        h0,  # N x D
        weight,  # D x 3D
        bias,  # 1 x 3D
        is_reverse,
        act_state,
        act_gate):
    def _seq_to_batch(lod, is_reverse):
        idx_in_seq_list = []
        seq_lens = lod[0]
        seq_starts = [0]
        for i in range(len(seq_lens)):
            seq_starts.append(seq_starts[-1] + seq_lens[i])
        sorted_seqs = sorted(
            list(range(len(seq_lens))),
            key=functools.cmp_to_key(lambda x, y: seq_lens[y] - seq_lens[x]))
        num_batch = seq_lens[sorted_seqs[0]]
        for batch_idx in range(num_batch):
            idx_in_seq = []
            for i in range(len(seq_lens)):
                if seq_lens[sorted_seqs[i]] <= batch_idx:
                    break
                idx = (seq_starts[sorted_seqs[i] + 1] - 1 - batch_idx
                       ) if is_reverse else (
                           seq_starts[sorted_seqs[i]] + batch_idx)
                idx_in_seq.append(idx)
            idx_in_seq_list.append(idx_in_seq)
        return idx_in_seq_list, sorted_seqs

    def _step(x, h_p, w, b, act_state, act_gate):
        T = x.shape[0]
        D = w.shape[0]
        g = x + np.tile(b, (T, 1))
        w_u_r = w.flatten()[:D * D * 2].reshape((D, D * 2))
        u_r = act_gate(np.dot(h_p, w_u_r) + g[:, :D * 2])
        u = u_r[:, :D]
        r = u_r[:, D:D * 2]
        r_h_p = r * h_p
        w_c = w.flatten()[D * D * 2:].reshape((D, D))
        c = act_state(np.dot(r_h_p, w_c) + g[:, D * 2:])
        g = np.hstack((u_r, c))
        h = u * c + (1 - u) * h_p
        return g, r_h_p, h

    T = sum(lod[0])
    N = len(lod[0])
    D = weight.shape[0]
    batch_gate = np.zeros((T, 3 * D), dtype='float64')
    batch_reset_hidden_prev = np.zeros((T, D), dtype='float64')
    batch_hidden = np.zeros((T, D), dtype='float64')
    hidden = np.zeros((T, D), dtype='float64')

    idx_in_seq_list, sorted_seqs = _seq_to_batch(lod, is_reverse)
    h_p = h0[sorted_seqs]
    max_seq_len = len(idx_in_seq_list)
    assert len(idx_in_seq_list[0]) == N
    end_idx = 0
    for batch_idx in range(max_seq_len):
        x = input[idx_in_seq_list[batch_idx]]
        g, r_h_p, h = _step(x, h_p, weight, bias, act_state, act_gate)
        if batch_idx < (max_seq_len - 1):
            h_p = h[:len(idx_in_seq_list[batch_idx + 1])]
        start_idx = end_idx
        end_idx = start_idx + len(idx_in_seq_list[batch_idx])
        batch_gate[start_idx:end_idx] = g
        batch_reset_hidden_prev[start_idx:end_idx] = r_h_p
        batch_hidden[start_idx:end_idx] = h
        hidden[idx_in_seq_list[batch_idx]] = h
    return batch_gate, batch_reset_hidden_prev, batch_hidden, hidden


class TestGRUOp(OpTest):
    def set_confs(self):
        pass

    def setUp(self):
        self.op_type = "gru"
        self.lod = [[2, 4, 3]]
        self.D = 5
        self.is_reverse = False
        self.with_h0 = True
        self.with_bias = True
        self.act_state = 'tanh'
        self.act_gate = 'sigmoid'
        self.set_confs()

        T = sum(self.lod[0])
        N = len(self.lod[0])

        input = np.random.rand(T, 3 * self.D).astype('float64')
        weight = np.random.rand(self.D, 3 * self.D).astype('float64')
        bias = np.random.rand(
            1, 3 * self.D).astype('float64') if self.with_bias else np.zeros(
                (1, 3 * self.D), dtype='float64')
        h0 = np.random.rand(
            N, self.D).astype('float64') if self.with_h0 else np.zeros(
                (N, self.D), dtype='float64')

        batch_gate, batch_reset_hidden_prev, batch_hidden, hidden = gru(
            input, self.lod, h0, weight, bias, self.is_reverse,
            ACTIVATION[self.act_state], ACTIVATION[self.act_gate])
        self.inputs = {'Input': (input, self.lod), 'Weight': weight}

        if self.with_bias:
            self.inputs['Bias'] = bias

        if self.with_h0:
            self.inputs['H0'] = h0

        self.outputs = {'Hidden': (hidden, self.lod)}

        self.attrs = {
            'activation': self.act_state,
            'gate_activation': self.act_gate,
            'is_reverse': self.is_reverse
        }

    def test_check_output(self):
        self.check_output(atol=1e-8)


class TestGRUOpNoInitial(TestGRUOp):
    def set_confs(self):
        self.with_h0 = False


class TestGRUOpNoBias(TestGRUOp):
    def set_confs(self):
        self.with_bias = False


class TestGRUOpReverse(TestGRUOp):
    def set_confs(self):
        self.is_reverse = True


if __name__ == "__main__":
    unittest.main()
