import torch
import torch.nn as nn


class SCRNCell(nn.Module):
    def __init__(self, input_size, num_units, context_units, alpha):
        super().__init__()
        self._input_size = input_size
        self._num_units = num_units
        self._context_units = context_units
        self._alpha = alpha
        self.B = nn.Parameter(torch.empty(input_size, context_units))
        self.V = nn.Parameter(torch.empty(context_units, num_units))
        self.U = nn.Parameter(torch.empty(num_units, num_units))
        self.fc = nn.Linear(context_units + input_size + num_units, num_units, bias=False)
        self.reset_parameters()  # weight initialization: glorot uniform

    # NOTE: rnn_cell_impl._linear: https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/rnn/python/ops/core_rnn_cell.py#L127
    def forward(self, inputs, state):
        # state_h.shape = (seq_len x batch_size x num_units), state_c.shape = (seq_len x batch_size x context_units)
        state_h, state_c = state.split([self._num_units, self._context_units], dim=2)
        # context_state.shape = (seq_len x batch_size x context_units)
        context_state = (1 - self._alpha) * (inputs @ self.B) + self._alpha * state_c
        # hidden_state.shape = (seq_len x batch_size x num_units)
        state_h = state_h.expand(inputs.shape[0], -1, -1)
        hidden_state = torch.sigmoid(self.fc(torch.cat([context_state, inputs, state_h], dim=2)))
        # output.shape = (seq_len x batch_size x num_units)
        output = hidden_state @ self.U + context_state @ self.V
        # new_state.shape = (seq_len x batch_size x (num_units+context_units))
        new_state = torch.cat([hidden_state, context_state], dim=2)
        return output, new_state

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight, gain=1.0)
