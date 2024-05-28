import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

        self.sos_id = 2
        self.eos_id = 3

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(vocab_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        # self.use_attention = use_attention
        self.attn_mode = attn_mode

        self.init_input = None
        self.masking = None
        self.input_dropout_p = input_dropout_p
        self.attention = use_attention
        # if use_attention:
        #    self.attention = Attention(self.hidden_size, attn_mode)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, regex, hidden, encoder_outputs, function):
        batch_size, regex_max_len = regex.shape

        regex = F.one_hot(regex.to(device="cuda"), num_classes=self.vocab_size).float()

        output, hidden = self.rnn(regex, hidden)
        attn = None
        self.attention.set_mask(self.masking)
        output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, regex_max_len, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax, masking=None, teacher_forcing_ratio=0.5):
        ret_dict = dict()
        ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        if masking is not None:
            self.masking = masking

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio)

        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    if self.attn_mode:
                        step_attn = ((attn[0][0][:, di, :, :], attn[0][1][:, di, :, :]), (attn[1][0][:, di, :], attn[1][1][:, di, :]))
                    else:  # attn only pos
                        step_attn = (attn[0][:, di, :, :], attn[1][:, di, :])
                else:
                    step_attn = None
                decode(di, step_output, step_attn)

        else:
            decoder_input = inputs[:, 0].unsqueeze(1)  # (batch, 1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, function=function)
                step_output = decoder_output.squeeze(1)
                if step_attn is not None:
                    if self.attn_mode:
                        step_attn = ((step_attn[0][0].squeeze(1), step_attn[0][1].squeeze(1)), (step_attn[1][0].squeeze(1), step_attn[1][1].squeeze(1)))
                    else:
                        step_attn = (step_attn[0].squeeze(1), step_attn[1].squeeze(1))
                else:
                    step_attn = None
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols

        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
