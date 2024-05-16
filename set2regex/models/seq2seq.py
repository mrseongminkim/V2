import torch.nn as nn
import torch.nn.functional as F

from models import EncoderRNN


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn1.flatten_parameters()
        if not self.encoder.set_transformer:
            self.encoder.rnn2.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, pos, neg, regex, teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden, masking = self.encoder(pos, neg)

        result = self.decoder(
            inputs=regex,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            function=self.decode_function,
            masking=masking,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return result
