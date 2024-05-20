import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention
from models.baseRNN import BaseRNN


class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = "attention_score"
    KEY_LENGTH = "length"
    KEY_SEQUENCE = "sequence"

    # except sos, eos
    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_size,
        n_layers=1,
        rnn_cell="LSTM",
        bidirectional=False,
        input_dropout_p=0,
        dropout_p=0,
        use_attention=False,
        attn_mode=False,
    ):
        super().__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional

        self.rnn = self.rnn_cell(vocab_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.attn_mode = attn_mode
        self.rnn1_hidden = None
        self.init_input = None
        self.masking = None
        self.input_dropout_p = input_dropout_p
        if use_attention:
            self.attention = Attention(self.hidden_size, attn_mode)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, pos, hidden, encoder_outputs, function):
        batch_size, n_examples, example_max_len = pos.shape

        pos = F.one_hot(pos.to(device="cuda"), num_classes=self.vocab_size).view(batch_size * n_examples, example_max_len, self.vocab_size).float()

        output, hidden = self.rnn(pos, hidden)

        attn = None
        if self.use_attention:
            self.attention.set_mask(self.masking)
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size * n_examples, example_max_len, self.output_size)

        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax, masking=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        if masking is not None:
            self.masking = masking

        batch_size = inputs.size(0)
        max_length = inputs.size(2)

        decoder_hidden = encoder_hidden

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        # step: single symbol index of regex, step_output = (640,12)
        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)
            return symbols

        decoder_output, decoder_hidden, attn = self.forward_step(inputs, decoder_hidden, encoder_outputs, function=function)

        for di in range(decoder_output.size(1)):
            step_output = decoder_output[:, di, :]
            if attn is not None:
                if self.attn_mode:
                    step_attn = ((attn[0][0][:, di, :, :], attn[0][1][:, di, :, :]), (attn[1][0][:, di, :], attn[1][1][:, di, :]))
                else:
                    step_attn = attn[:, di, :]
            else:
                step_attn = None
            decode(di, step_output, step_attn)

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols

        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict
