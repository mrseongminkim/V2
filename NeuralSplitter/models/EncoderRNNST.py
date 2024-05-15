import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN
from string_preprocess import get_mask

from models.set_transformer.model import SetTransformer


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


class EncoderRNNST(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Applies a set transformer

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    """

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
    ):
        super().__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.vocab_size = vocab_size
        self.embed_size = 4

        self.variable_lengths = variable_lengths
        self.vocab = vocab
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.input_dropout_p = input_dropout_p
        self.n_layers = n_layers
        self.rnn1 = self.rnn_cell(
            self.vocab_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # dropout=dropout_p,
        )

        self.set_transformer = SetTransformer(hidden_size * 2, hidden_size * 2, n_layers)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        self.encoder_1 = nn.Sequential(
            SAB(dim_in=hidden_size * 2, dim_out=hidden_size * 2, num_heads=4, ln=True),
            SAB(dim_in=hidden_size * 2, dim_out=hidden_size * 2, num_heads=4, ln=True),
        )
        self.decoder_1 = nn.Sequential(
            PMA(dim=hidden_size * 2, num_heads=4, num_seeds=1, ln=True),
            nn.Linear(
                in_features=hidden_size * 2,
                out_features=hidden_size * 2,
            ),
        )

    def forward(self, input_var, input_lengths=None, embedding=None):
        batch_size, set_size, seq_len = input_var.size(0), input_var.size(1), input_var.size(2)
        one_hot = F.one_hot(input_var.to(device="cuda"), num_classes=self.vocab_size)
        src_embedded = one_hot.view(batch_size * set_size, seq_len, -1).float()

        masking = get_mask(input_var)  # batch, set_size, seq_len

        src_output, src_hidden = self.rnn1(src_embedded)  # (batch x set_size, seq_len, hidden), # (num_layer x num_dir, batch*set_size, hidden)

        rnn1_hidden = src_hidden

        src_output = src_output.view(batch_size, set_size, seq_len, -1)  # batch, set_size, seq_len, hidden)

        if type(self.rnn1) is nn.LSTM:
            src_single_hidden = src_hidden[0].view(self.n_layers, -1, batch_size * set_size, self.hidden_size)  # num_layer(2), num_direction, batch x set_size, hidden
        else:
            src_single_hidden = src_hidden.view(self.n_layers, -1, batch_size * set_size, self.hidden_size)  # num_layer(2), num_direction, batch x set_size, hidden

        # use hidden state of final_layer
        set_embedded = src_single_hidden[-1, :, :, :]  # num_direction, batch x set_size, hidden

        if self.bidirectional:
            set_embedded = torch.cat((set_embedded[0], set_embedded[1]), dim=-1)  # batch x set_size, num_direction x hidden
        else:
            set_embedded = set_embedded.squeeze(0)  # batch x set_size, hidden

        set_embedded = set_embedded.view(batch_size, set_size, -1)  # batch, set_size, hidden
        # set_embedded: batch, n_examples, hidden * 2

        encoder_1 = self.encoder_1(set_embedded)

        decoder_1 = self.decoder_1(encoder_1).squeeze(1)  # .unsqueeze(0)

        set_hidden = self.batch_norm(decoder_1).unsqueeze(0)

        set_hidden = set_hidden.repeat_interleave(self.n_layers, dim=0)

        # set_hidden = set_hidden.view(self.n_layers * batch_size, -1)
        # set_hidden = self.batch_norm(set_hidden)
        # set_hidden = set_hidden.view(self.n_layers, batch_size, -1)

        return (src_output, None), set_hidden, masking, rnn1_hidden


# Encoder와 차이점: rnn2 -> set-transformer and batch-norm
