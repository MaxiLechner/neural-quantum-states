from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.experimental import stax
from jax.experimental.stax import Relu
from masked_conv_layer import MaskedConv1d


def net(M, FilterSize):
    Main = stax.serial(
        MaskedConv1d(M, (FilterSize,), 0, padding="SAME"),
        Relu,
        MaskedConv1d(M, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(M, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(M, (FilterSize,), padding="SAME"),
    )
    return Main
