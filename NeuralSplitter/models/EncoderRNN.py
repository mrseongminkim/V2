import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseRNN import BaseRNN
from string_preprocess import get_mask


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class EncoderRNN(BaseRNN):
    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_size,
        input_dropout_p=0,
        dropout_p=0,
        n_layers=1,
        bidirectional=False,
        rnn_cell="LSTM",
        variable_lengths=False,
        vocab=None,
        set_transformer=False,
    ):
        super().__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.set_transformer = set_transformer

        self.vocab_size = vocab_size
        self.variable_lengths = variable_lengths
        self.vocab = vocab
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.input_dropout_p = input_dropout_p
        self.n_layers = n_layers
        self.n_directions = 2 if self.bidirectional else 1

        self.rnn1 = self.rnn_cell(
            self.vocab_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if set_transformer:
            self.rnn2 = nn.Sequential(
                SAB(dim_in=hidden_size * 2, dim_out=hidden_size * 2, num_heads=8, ln=True),
                SAB(dim_in=hidden_size * 2, dim_out=hidden_size * 2, num_heads=8, ln=True),
                PMA(dim=hidden_size * 2, num_heads=8, num_seeds=1, ln=True),
                nn.Linear(
                    in_features=hidden_size * 2,
                    out_features=hidden_size * 2,
                ),
            )
        else:
            self.rnn2 = self.rnn_cell(
                hidden_size * 2 if self.bidirectional else hidden_size,
                hidden_size,
                n_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_p,
            )

        self.hidden_linear = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.cell_linear = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)

    def _cat_directions(self, h):
        """If the encoder is bidirectional, do the following transformation.
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional:
            h = torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], 2)
        return h

    def forward(self, pos):
        batch_size, n_examples, example_max_len = pos.shape

        masking = get_mask(pos)

        pos = F.one_hot(pos.to(device="cuda"), num_classes=self.vocab_size).view(batch_size * n_examples, example_max_len, self.vocab_size).float()

        pos_output, pos_hidden = self.rnn1(pos)

        pos_output = pos_output.view(batch_size, n_examples, example_max_len, self.hidden_size * 2)

        if type(self.rnn1) is nn.LSTM:
            pos_set = pos_hidden[0].view(self.n_layers, self.n_directions, batch_size * n_examples, self.hidden_size)
        else:
            pos_set = pos_hidden.view(self.n_layers, self.n_directions, batch_size * n_examples, self.hidden_size)
        pos_set = pos_set[-1, :, :, :]

        if self.bidirectional:
            pos_set = torch.cat((pos_set[0], pos_set[1]), dim=-1)
        else:
            pos_set = pos_set.squeeze(0)

        pos_set = pos_set.view(batch_size, n_examples, self.hidden_size * 2)

        if self.set_transformer:
            set_hidden = self.rnn2(pos_set).squeeze(1).unsqueeze(0).repeat_interleave(2, dim=0).repeat_interleave(10, dim=1)
            if type(self.rnn1) is nn.LSTM:
                set_cell = torch.zeros_like(set_hidden)

                pos_cell = self._cat_directions(pos_hidden[1])
                pos_hidden = self._cat_directions(pos_hidden[0])

                hidden = torch.cat((pos_hidden, set_hidden), dim=-1)
                cell = torch.cat((pos_cell, set_cell), dim=-1)

                hidden = self.hidden_linear(hidden)
                cell = self.cell_linear(cell)

                hidden = (hidden, cell)
            else:
                pos_hidden = self._cat_directions(pos_hidden)

                hidden = torch.cat((pos_hidden, set_hidden), dim=-1)
                hidden = self.hidden_linear(hidden)
        else:
            _, set_hidden = self.rnn2(pos_set)
            if type(self.rnn1) is nn.LSTM:
                set_cell = self._cat_directions(set_hidden[1]).repeat_interleave(10, dim=1)
                set_hidden = self._cat_directions(set_hidden[0]).repeat_interleave(10, dim=1)

                pos_cell = self._cat_directions(pos_hidden[1])
                pos_hidden = self._cat_directions(pos_hidden[0])

                hidden = torch.cat((pos_hidden, set_hidden), dim=-1)
                cell = torch.cat((pos_cell, set_cell), dim=-1)

                hidden = self.hidden_linear(hidden)
                cell = self.cell_linear(cell)

                hidden = (hidden, cell)
            else:
                set_hidden = self._cat_directions(set_hidden).repeat_interleave(10, dim=1)

                pos_hidden = self._cat_directions(pos_hidden)

                hidden = torch.cat((pos_hidden, set_hidden), dim=-1)

                hidden = self.hidden_linear(hidden)

        outputs = pos_output
        hiddens = hidden
        masking = masking

        return outputs, hiddens, masking
