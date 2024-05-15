import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, rnn_type, set_transformer):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_directions = 2
        if rnn_type == "lstm":
            self.rnn = nn.LSTM
        elif rnn_type == "gru":
            self.rnn = nn.GRU
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        self.rnn_type = rnn_type
        self.example_rnn = self.rnn(vocab_size, hidden_dim, n_layers, bidirectional=True)
        self.set_rnn = self.rnn(hidden_dim * 2, hidden_dim, n_layers, bidirectional=True)
        self.hidden_linear = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.cell_linear = nn.Linear(hidden_dim * 4, hidden_dim * 2)

    def concat_directions(self, h):
        return torch.cat((h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]), dim=-1)

    def forward(self, pos):
        example = 10
        batch = pos.shape[1] // example
        # pos: example_length, batch * example

        pos = F.one_hot(pos, num_classes=self.vocab_size).float()
        # pos: example_length, batch * example, vocab_size

        pos_output, pos_hidden = self.example_rnn(pos)
        # pos_output: example_length, batch * example, hidden_dim * 2
        # pos_hidden: n_layers * n_directions, batch * example, hidden_dim

        if self.rnn_type == "lstm":
            pos_set = torch.cat((pos_hidden[0][-2], pos_hidden[0][-1]), dim=-1)
        else:
            pos_set = torch.cat((pos_hidden[-2], pos_hidden[-1]), dim=-1)
        pos_set = pos_set.view(example, batch, self.hidden_dim * 2)

        # Set-Transformer import points
        pos_set_outputs, pos_set_hidden = self.set_rnn(pos_set)
        # pos_set_outputs: example, batch, hidden_dim * 2
        # pos_set_hidden: n_layers * n_directions, batch, hidden_dim

        if self.rnn_type == "lstm":
            set_hidden = self.concat_directions(pos_set_hidden[0]).repeat_interleave(example, dim=1)
            set_cell = self.concat_directions(pos_set_hidden[1]).repeat_interleave(example, dim=1)

            string_hidden = self.concat_directions(pos_hidden[0])
            string_cell = self.concat_directions(pos_hidden[1])
            hidden = torch.cat((set_hidden, string_hidden), dim=-1)
            cell = torch.cat((set_cell, string_cell), dim=-1)

            hidden = self.hidden_linear(hidden)
            cell = self.cell_linear(cell)

            hidden = (hidden, cell)
        else:
            pos_set_hidden = self.concat_directions(pos_set_hidden).repeat_interleave(example, dim=1)
            pos_hidden = self.concat_directions(pos_hidden)
            hidden = torch.cat((pos_set_hidden, pos_hidden), dim=-1)
            hidden = self.hidden_linear(hidden)
        # pos_output: example_length, batch * example, hidden_dim * 2
        # hidden: n_layers, batch * example, hidden_dim * 2
        # cell: n_layers, batch * example, hidden_dim * 2

        return pos_output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, output, context):
        batch_size = output.size(0)
        example_len = output.size(1)
        hidden_size = output.size(2)

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn.view(-1, example_len), dim=1).view(batch_size, example_len, example_len)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, rnn_type):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim * 2
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(vocab_size, self.hidden_dim, n_layers)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(vocab_size, self.hidden_dim, n_layers)
        self.attention = Attention(self.hidden_dim)
        self.linear = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, pos, pos_output, hidden):
        example = 10
        batch = pos.shape[1] // example

        pos = F.one_hot(pos, num_classes=self.vocab_size).float()
        # pos: example_length, batch * example, vocab_size

        output, hidden = self.rnn(pos, hidden)
        # output: example_length, batch * example, hidden_dim * 2
        # hidden: n_layers, batch * example, hidden_dim * 2

        output, attention = self.attention(output.permute(1, 0, 2), pos_output.permute(1, 0, 2))
        # output: batch * example, example_length, hiddem_dim * 2
        # attention: batch * example, example_length, example_length

        output = output.permute(1, 0, 2)

        output = self.linear(output)

        return output, hidden, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, pos):
        # pos: example_length, batch * example

        pos_output, hidden = self.encoder(pos)

        output, hidden, attention = self.decoder(pos, pos_output, hidden)

        return output
