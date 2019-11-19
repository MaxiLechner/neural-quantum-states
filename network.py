from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.experimental import stax
from jax.experimental.stax import Relu, FanOut, FanInSum, Identity
from masked_conv_layer import MaskedConv1d


def resnet_block_1d(width, FilterSize):
    Main = stax.serial(
        MaskedConv1d(width, FilterSize, padding="SAME"),
        Relu,
        MaskedConv1d(width, FilterSize, padding="SAME"),
        Relu,
        MaskedConv1d(width, FilterSize, padding="SAME"),
    )
    Shortcut = Identity
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def small_resnet_1d(width, FilterSize):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), 0, padding="SAME"),
        Relu,
        resnet_block_1d(width, (FilterSize,)),
        # resnetBlock(12, (FilterSize,)),
        MaskedConv1d(4, (1,), padding="SAME"),
        Relu,
    )
    return Main


def small_net_1d(width, FilterSize):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), 0, padding="SAME"),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(4, (FilterSize,), padding="SAME"),
    )
    return Main
