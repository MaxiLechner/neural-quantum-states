from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.experimental import stax
from jax.experimental.stax import Relu, FanOut, FanInSum, Identity
from masked_conv_layer import MaskedConv1d


def resnetBlock(M, FilterSize):
    Main = stax.serial(
        MaskedConv1d(M, FilterSize, padding="SAME"),
        Relu,
        MaskedConv1d(M, FilterSize, padding="SAME"),
        Relu,
        MaskedConv1d(M, FilterSize, padding="SAME"),
    )
    Shortcut = Identity
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def resnet(M, FilterSize):
    Main = stax.serial(
        MaskedConv1d(12, (FilterSize,), 0, padding="SAME"),
        Relu,
        resnetBlock(12, (FilterSize,)),
        # resnetBlock(12, (FilterSize,)),
        MaskedConv1d(M, (1,), padding="SAME"),
        Relu,
    )
    return Main


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
