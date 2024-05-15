import torch
import torch.nn as nn
from .blocks import SetAttentionBlock
from .blocks import InducedSetAttentionBlock
from .blocks import PoolingMultiheadAttention


class SetTransformer(nn.Module):

    def __init__(self, in_dimension, out_dimension, n_layers):
        """
        Arguments:
            in_dimension: an integer.
            out_dimension: an integer.
        """
        super().__init__()

        self.n_layers = n_layers

        d = 128
        m = 10  # number of inducing points
        h = 4  # number of heads
        k = 10  # number of seed vectors

        self.embed = nn.Sequential(nn.Linear(in_dimension, d), nn.ReLU(inplace=True))
        self.encoder_1 = nn.Sequential(InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)), InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)))
        self.encoder_2 = nn.Sequential(InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)), InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)))

        self.decoder_1 = nn.Sequential(PoolingMultiheadAttention(d, k, h, RFF(d)), SetAttentionBlock(d, h, RFF(d)))
        self.decoder_2 = nn.Sequential(PoolingMultiheadAttention(d, k, h, RFF(d)), SetAttentionBlock(d, h, RFF(d)))

        self.predictor_1 = nn.Linear(k * d, out_dimension)
        self.predictor_2 = nn.Linear(k * d, out_dimension)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, in_dimension].
        Returns:
            a float tensor with shape [b, out_dimension].
        """
        x = self.embed(x)  # shape [b, n, d]

        encoder_1 = self.encoder_1(x)
        decoder_1 = self.decoder_1(encoder_1)
        b, k, d = decoder_1.shape
        decoder_1 = decoder_1.view(b, k * d)
        predictor_1 = self.predictor_1(decoder_1)

        if self.n_layers == 1:
            predictor_1 = predictor_1.unsqueeze(0)
            return predictor_1
        else:
            encoder_2 = self.encoder_2(encoder_1)
            decoder_2 = self.decoder_2(encoder_2)
            b, k, d = decoder_2.shape
            decoder_2 = decoder_2.view(b, k * d)
            predictor_2 = self.predictor_2(decoder_2)

            hidden = torch.cat((predictor_1.unsqueeze(0), predictor_2.unsqueeze(0)), dim=0)

            return hidden


class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """

    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)
