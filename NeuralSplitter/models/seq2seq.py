import torch.nn as nn
import torch.nn.functional as F

from models import EncoderRNN


class Seq2seq(nn.Module):
    """Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.embed_size = 4
        self.shared_embedding = nn.Embedding(encoder.vocab_size, self.embed_size)

    def flatten_parameters(self):
        self.encoder.rnn1.flatten_parameters()
        if isinstance(self.encoder, EncoderRNN):
            self.encoder.rnn2.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None, teacher_forcing_ratio=0):

        encoder_outputs, encoder_hidden, masking, rnn1_hidden = self.encoder(input_variable, input_lengths, self.shared_embedding)

        result = self.decoder(
            inputs=input_variable,
            embedding=self.shared_embedding,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            function=self.decode_function,
            teacher_forcing_ratio=teacher_forcing_ratio,
            masking=masking,
            rnn1_hidden=rnn1_hidden,
        )

        return result
