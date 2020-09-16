from typing import List

import torch
import torch.nn as nn


class MultiplicativeIntegration(nn.Module):
    def __init__(self,
                 inputs_sizes: List[int],
                 output_sizes: List[int],
                 bias: bool,
                 bias_start: float = 0.0,
                 alpha_start: float = 1.0,
                 beta_start: float = 1.0):
        super().__init__()
        self.inputs_sizes = inputs_sizes
        self.output_sizes = output_sizes
        total_output_size = sum(output_sizes)
        total_input_size = sum(inputs_sizes)
        self.bias_start = bias_start
        self.alpha_start = alpha_start
        self.beta_start = beta_start
        self.weights = nn.Parameter(torch.empty(total_input_size, total_output_size))
        self.alphas = nn.Parameter(torch.empty([total_output_size]))
        self.betas = nn.Parameter(torch.empty([2*total_output_size]))
        self.biases = nn.Parameter(torch.empty([total_output_size])) if bias else None
        self.reset_parameters()

    def forward(self, input0, input1):
        # input0.shape = (seq_len x batch_size x input_size), input1.shape = (seq_len x batch_size x num_units)
        # w1.shape = (input_size x 4 * num_units), w2.shape = (num_units x 4 * num_units)
        w1, w2 = torch.split(self.weights, self.inputs_sizes, dim=0)
        # b1.shape, b2.shape = (4 * num_units)
        b1, b2 = torch.split(self.betas, sum(self.output_sizes), dim=0)
        # wx1.shape = (seq_len x batch_size x 4 * num_units), wx2.shape = (seq_len x batch_size x 4 * num_units)
        wx1, wx2 = input0 @ w1, input1 @ w2
        # res.shape = (seq_len x batch_size x 4 * num_units)
        res = self.alphas * wx1 * wx2 + b1 * wx1 + b2 * wx2
        if self.biases is not None: res += self.biases
        return res

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=1.0)
        nn.init.constant_(self.alphas, self.alpha_start)
        nn.init.constant_(self.betas, self.beta_start)
        if self.biases is not None:
            nn.init.constant_(self.biases, self.bias_start)


class MILSTMCell(nn.Module):
    def __init__(self, input_size, num_units, forget_bias=0.0,
                 bias_start=0.0, alpha_start=1.0,
                 beta_start=1.0, activation=torch.tanh):
        super().__init__()
        self._input_size = input_size
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._bias_start = bias_start
        self._alpha_start = alpha_start
        self._beta_start = beta_start
        self._activation = activation
        self.mi_module = MultiplicativeIntegration(
            inputs_sizes=[input_size, num_units],
            output_sizes=[num_units, num_units, num_units, num_units],
            bias=True,
            bias_start=bias_start,
            alpha_start=alpha_start,
            beta_start=beta_start,
        )

    def forward(self, inputs, state):
        # c/h.shape = (seq_len x batch_size x num_units)
        c, h = state
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        concat = self.mi_module(inputs, h)
        # i/j/f/o.shape = (seq_len x batch_size x num_units)
        i, j, f, o = torch.split(concat, self._num_units, dim=2)
        # new_c.shape = (seq_len x batch_size x num_units)
        new_c = c * torch.sigmoid(f + self._forget_bias) + torch.sigmoid(i + self._activation(j))
        # new_h.shape = (seq_len x batch_size x num_units)
        new_h = self._activation(new_c) * torch.sigmoid(o)
        new_state = new_c, new_h
        return new_h, new_state
