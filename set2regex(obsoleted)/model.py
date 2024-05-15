import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, rnn_type):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_directions = 2

        self.rnn_type = rnn_type
        if self.rnn_type == "lstm":
            self.example_rnn = nn.LSTM(vocab_size, hidden_dim, n_layers, bidirectional=True)
            self.set_rnn = nn.LSTM(hidden_dim * 2, hidden_dim * 2, n_layers, bidirectional=True)
        elif self.rnn_type == "gru":
            self.example_rnn = nn.GRU(vocab_size, hidden_dim, n_layers, bidirectional=True)
            self.set_rnn = nn.GRU(hidden_dim * 2, hidden_dim * 2, n_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2 * 2 * 2, hidden_dim * 2)

    def concat_directions(self, h):
        return torch.cat((h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]), dim=-1)

    def forward(self, pos, neg):
        example = 10
        batch = pos.shape[1] // example
        # pos: example_length, batch * example
        # neg: example_length, batch * example

        pos = F.one_hot(pos, num_classes=self.vocab_size).float()
        neg = F.one_hot(neg, num_classes=self.vocab_size).float()
        # pos: example_length, batch * example, vocab_size
        # neg: example_length, batch * example, vocab_size

        if self.rnn_type == "lstm":
            pos_outputs, (pos_hidden, pos_cell) = self.example_rnn(pos)
            neg_outputs, (neg_hidden, neg_cell) = self.example_rnn(pos)
            # _outputs: example_length, batch * example, hidden_dim * 2
            # _hidden: n_layers * n_directions, batch * example, hidden_dim
            # _cell: n_layers * n_directions, batch * example, hidden_dim

            pos_set = torch.cat((pos_hidden[-2], pos_hidden[-1]), dim=-1).view(example, batch, self.hidden_dim * 2)
            neg_set = torch.cat((neg_hidden[-2], neg_hidden[-1]), dim=-1).view(example, batch, self.hidden_dim * 2)
            # _set: example, batch, hidden_dim * 2

            pos_set_outputs, (pos_set_hidden, pos_set_cell) = self.set_rnn(pos_set)
            neg_set_outputs, (neg_set_hidden, neg_set_cell) = self.set_rnn(neg_set)
            # _set_outputs: example, batch, hidden_dim * 2 * 2
            # _set_hidden: n_layers * n_directions, batch, hidden_dim * 2
            # _set_cell: n_layers * n_directions, batch, hidden_dim * 2

            pos_set_hidden = self.concat_directions(pos_set_hidden)
            neg_set_hidden = self.concat_directions(neg_set_hidden)
            pos_set_cell = self.concat_directions(pos_set_cell)
            neg_set_cell = self.concat_directions(neg_set_cell)
            # _set_hidden: n_layers, batch, hidden_dim * 2 * 2
            # _set_cell: n_layers, batch, hidden_dim * 2 * 2

            set_outputs = torch.cat((pos_set_outputs, neg_set_outputs), dim=0)
            set_hidden = torch.cat((pos_set_hidden, neg_set_hidden), dim=-1)
            set_cell = torch.cat((pos_set_cell, neg_set_cell), dim=-1)
            # set_outputs: example * 2, batch, hidden_dim * 2 * 2
            # set_hidden: n_layers, batch, hidden_dim * 2 * 2 * 2
            # set_cell: n_layers, batch, hidden_dim * 2 * 2 * 2

            set_hidden = torch.tanh(self.linear(set_hidden))
            set_cell = torch.tanh(self.linear(set_cell))
            # set_hidden: n_layers, batch, hidden_dim * 2
            # set_cell: n_layers, batch, hidden_dim * 2

            return set_outputs, set_hidden, set_cell

        elif self.rnn_type == "gru":
            pos_outputs, pos_hidden = self.example_rnn(pos)
            neg_outputs, neg_hidden = self.example_rnn(neg)
            # _outputs: example_length, batch * example, hidden_dim * 2
            # _hidden: n_layers * n_directions, batch * example, hiddem_dim

            pos_set = torch.cat((pos_hidden[-2], pos_hidden[-1]), dim=-1).view(example, batch, self.hidden_dim * 2)
            neg_set = torch.cat((neg_hidden[-2], neg_hidden[-1]), dim=-1).view(example, batch, self.hidden_dim * 2)
            # _set: example, batch, hidden_dim * 2

            pos_set_outputs, pos_set_hidden = self.set_rnn(pos_set)
            neg_set_outputs, neg_set_hidden = self.set_rnn(neg_set)
            # _set_outputs: example, batch, hiddem_dim * 2 * 2
            # _set_hidden: n_layers * n_directions, batch, hidden_dim * 2

            pos_set_hidden = self.concat_directions(pos_set_hidden)
            neg_set_hidden = self.concat_directions(neg_set_hidden)
            # _set_hidden: n_layers, batch, hidden_dim * 2 * 2

            set_outputs = torch.cat((pos_set_outputs, neg_set_outputs), dim=0)
            set_hidden = torch.cat((pos_set_hidden, neg_set_hidden), dim=-1)
            # set_outputs: example * 2, batch, hidden_dim * 2 * 2
            # set_hidden: n_layers, batch, hidden_dim * 2 * 2 * 2

            set_hidden = torch.tanh(self.linear(set_hidden))
            # set_hidden: n_layers, batch, hidden_dim * 2

            return set_outputs, set_hidden, None


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(hidden_dim * 2 * 2 + hidden_dim * 2, hidden_dim * 2)
        self.v_fc = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: batch, hidden_dim * 2
        # encoder_outputs: example * 2, batch, hidden_dim * 2 * 2

        batch_size = encoder_outputs.shape[1]
        n_examples = encoder_outputs.shape[0]

        # repeat decoder hidden state example * 2 times
        hidden = hidden.unsqueeze(1).repeat(1, n_examples, 1)
        # hidden: batch, examplle * 2, hidden_dim * 2

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs: batch, example * 2, hidden_dim * 2 * 2

        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=-1)))
        # energy: batch, example * 2, hidden_dim * 2

        attention = self.v_fc(energy).squeeze(2)
        # attention: batch, example * 2
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, rnn_type, attention):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim * 2
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.attention = attention
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_dim * 4 + vocab_size, self.hidden_dim, n_layers)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(hidden_dim * 4 + vocab_size, self.hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim * 4 + hidden_dim * 2 + vocab_size, vocab_size)

    def forward(self, input, encoder_outputs, hidden, cell):
        # input: batch
        # encoder_outputs: example * 2, batch, hidden_dim * 2 * 2
        # hidden: n_layers, batch, hidden_dim * 2
        # cell: n_layers, batch, hidden_dim * 2

        input = input.unsqueeze(0)
        # input: 1, batch size

        input = F.one_hot(input, num_classes=self.vocab_size).float()
        # input: 1, batch size, vocab_size

        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        # a: batch, 1, example * 2

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs: batch, example * 2, hiddem_dim * 2 * 2

        weighted = torch.bmm(a, encoder_outputs)
        # weighted: batch, 1, hiddem_dim * 2 * 2

        weighted = weighted.permute(1, 0, 2)
        # weighted: 1, batch, hiddem_dim * 2 * 2

        rnn_input = torch.cat((input, weighted), dim=-1)
        # rnn_input: 1, batch, hidden_dim * 2 * 2 + vocab_size

        if self.rnn_type == "lstm":
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            # output: 1, batch, hidden_dim * 2
            # hidden: n_layers * n_directions, batch, hidden * 2
            # cell: n_layers * n_directions, batch, hidden * 2

            input = input.squeeze(0)
            output = output.squeeze(0)
            weighted = weighted.squeeze(0)
            prediction = self.linear(torch.cat((output, weighted, input), dim=-1))

            return prediction, hidden, cell, a.squeeze(1)

        elif self.rnn_type == "gru":
            output, hidden = self.rnn(rnn_input, hidden)
            # output: 1, batch, hidden_dim * 2
            # hidden: n_layers, 1, batch, hidden_dim * 2

            input = input.squeeze(0)
            output = output.squeeze(0)
            weighted = weighted.squeeze(0)
            prediction = self.linear(torch.cat((output, weighted, input), dim=-1))

            return prediction, hidden, None, a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, rnn_type):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.rnn_type = rnn_type

    def forward(self, pos, neg, regex, teacher_forcing_ratio):
        # pos: example_length, batch * example
        # neg: example_lenght, batch * example
        # regex: regex_length, batch
        regex_length, batch = regex.shape
        vocab_size = self.encoder.vocab_size

        # tensor to store decoder outputs
        outputs = torch.zeros(regex_length, batch, vocab_size).to(self.device)
        # outputs: regex_length, batch, vocab_size

        if self.rnn_type == "lstm":
            # last hidden state of the encoder is used as the initial hidden state of the decoder
            encoder_outputs, hidden, cell = self.encoder(pos, neg)
            # set_outputs: example * 2, batch, hidden_dim * 2 * 2
            # set_hidden: n_layers, batch, hidden_dim * 2
            # set_cell: n_layers, batch, hidden_dim * 2
        elif self.rnn_type == "gru":
            encoder_outputs, hidden, cell = self.encoder(pos, neg)

        # first input to the decoder is the <sos> tokens
        input = regex[0, :]
        # input = [batch size]
        for t in range(1, regex_length):
            output, hidden, cell, attention = self.decoder(input, encoder_outputs, hidden, cell)
            # prediction: batch, vocab_size
            # hidden: n_layers, batch, hidden * 2
            # cell: n_layers, batch, hidden * 2

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = regex[t] if teacher_force else top1
            # input: batch
        return outputs
