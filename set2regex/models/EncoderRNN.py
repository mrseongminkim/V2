import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseRNN import BaseRNN
from string_preprocess import get_mask


class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

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
        max_len,  # 이거 안 쓰임
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
            dropout=dropout_p,
        )

        self.rnn2 = self.rnn_cell(
            hidden_size * 2 if self.bidirectional else hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p,
        )

    def forward(self, pos, neg, input_lengths=None, embedding=None):
        batch_size, set_size, seq_len = pos.size(0), pos.size(1), pos.size(2)

        pos_one_hot = F.one_hot(pos.to(device="cuda"), num_classes=self.vocab_size)
        neg_one_hot = F.one_hot(neg.to(device="cuda"), num_classes=self.vocab_size)

        # src_embedded = embedding(input_var.reshape(batch_size * set_size, seq_len))
        pos_src_embedded = pos_one_hot.view(batch_size * set_size, seq_len, -1).float()
        neg_src_embedded = neg_one_hot.view(batch_size * set_size, seq_len, -1).float()

        masking = get_mask(pos)  # batch, set_size, seq_len

        pos_src_output, pos_src_hidden = self.rnn1(pos_src_embedded)  # (batch x set_size, seq_len, hidden), # (num_layer x num_dir, batch*set_size, hidden)
        neg_src_output, neg_src_hidden = self.rnn1(neg_src_embedded)

        # 모든 time step에 대해서 정보를 가지고 있다.
        pos_src_output = pos_src_output.view(batch_size, set_size, seq_len, -1)  # batch, set_size, seq_len, hidden)
        neg_src_output = neg_src_output.view(batch_size, set_size, seq_len, -1)

        if type(self.rnn1) is nn.LSTM:
            pos_src_single_hidden = pos_src_hidden[0].view(self.n_layers, -1, batch_size * set_size, self.hidden_size)  # num_layer(2), num_direction, batch x set_size, hidden
            neg_src_single_hidden = neg_src_hidden[0].view(self.n_layers, -1, batch_size * set_size, self.hidden_size)
        else:
            pos_src_single_hidden = pos_src_hidden.view(self.n_layers, -1, batch_size * set_size, self.hidden_size)  # num_layer(2), num_direction, batch x set_size, hidden
            neg_src_single_hidden = neg_src_hidden.view(self.n_layers, -1, batch_size * set_size, self.hidden_size)
        # https://stackoverflow.com/questions/56677052/is-hidden-and-output-the-same-for-a-gru-unit-in-pytorch

        # use hidden state of final_layer
        pos_set_embedded = pos_src_single_hidden[-1, :, :, :]  # num_direction, batch x set_size, hidden
        neg_set_embedded = neg_src_single_hidden[-1, :, :, :]

        if self.bidirectional:
            pos_set_embedded = torch.cat((pos_set_embedded[0], pos_set_embedded[1]), dim=-1)  # batch x set_size, num_direction x hidden
            neg_set_embedded = torch.cat((neg_set_embedded[0], neg_set_embedded[1]), dim=-1)
        else:
            pos_set_embedded = pos_set_embedded.squeeze(0)  # batch x set_size, hidden
            neg_set_embedded = neg_set_embedded.squeeze(0)

        pos_set_embedded = pos_set_embedded.view(batch_size, set_size, -1)  # batch, set_size, hidden
        neg_set_embedded = neg_set_embedded.view(batch_size, set_size, -1)

        # 여기서 셋 트랜스포머를 사용해야한다.
        # 모든 시점에서의 정보를 다 가지고 있다.
        pos_set_output, pos_set_hidden = self.rnn2(pos_set_embedded)  # (batch, set_size, hidden), # (num_layer*num_dir, batch, hidden) 2개 tuple 구성
        neg_set_output, neg_set_hidden = self.rnn2(neg_set_embedded)

        outputs = torch.cat((pos_set_output, neg_set_output), dim=1)

        if type(self.rnn2) is nn.LSTM:
            pass
            # last_hidden = set_hidden[0]  # num_layer x num_dir, batch, hidden
            # last_cell = set_hidden[1]  # num_layer x num_dir, batch, hidden
            # hiddens = (last_hidden, last_cell)
        else:
            hiddens = torch.cat((pos_set_hidden, neg_set_hidden), dim=-1)  # num_layer x num_dir, batch, hidden

        return outputs, hiddens, masking
