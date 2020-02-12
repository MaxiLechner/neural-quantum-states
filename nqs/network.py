from jax import lax
import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import Relu, FanOut, FanInSum, Identity, randn, glorot

import itertools


def MaskedConv1d(
    out_chan,
    filter_shape,
    is_first_layer=False,
    strides=None,
    padding="VALID",
    W_init=None,
    b_init=randn(1e-6),
):
    """Layer construction function for a 1d masked convolution layer."""
    assert len(filter_shape) == 1, "filter_shape must be one dimensional"
    assert filter_shape[0] % 2 != 0, "filter_shape must be odd"

    def compute_mask(is_first_layer):
        if is_first_layer:
            return np.array(
                [
                    [[1]] if i < (filter_shape[0] - 1) / 2 else [[0]]
                    for i in range(filter_shape[0])
                ]
            )
        else:
            return np.array(
                [
                    [[1]] if i < (filter_shape[0] + 1) / 2 else [[0]]
                    for i in range(filter_shape[0])
                ]
            )

    mask = compute_mask(is_first_layer)
    dimension_numbers = ("NWC", "WIO", "NWC")
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or glorot(rhs_spec.index("O"), rhs_spec.index("I"))

    def init_fun(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [
            out_chan
            if c == "O"
            else input_shape[lhs_spec.index("C")]
            if c == "I"
            else next(filter_shape_iter)
            for c in rhs_spec
        ]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers
        )
        bias_shape = [out_chan if c == "C" else 1 for c in out_spec]
        bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
        W, b = (
            W_init(rng, kernel_shape, dtype=np.float64),
            b_init(rng, bias_shape, dtype=np.float64),
        )
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return (
            lax.conv_general_dilated(
                inputs, W * mask, strides, padding, one, one, dimension_numbers
            )
            + b
        )

    return init_fun, apply_fun


def resnet_block_1d(width, FilterSize):
    Main = stax.serial(
        MaskedConv1d(width, FilterSize, padding="SAME"),
        Relu,
        MaskedConv1d(width, FilterSize, padding="SAME"),
        Relu,
        MaskedConv1d(width, FilterSize, padding="SAME"),
    )
    Shortcut = Identity
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum)


def small_resnet_1d(width, FilterSize):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, padding="SAME"),
        Relu,
        resnet_block_1d(width, (FilterSize,)),
        Relu,
        resnet_block_1d(width, (FilterSize,)),
        Relu,
        resnet_block_1d(width, (FilterSize,)),
        Relu,
        MaskedConv1d(4, (FilterSize,), padding="SAME"),
    )
    return Main


def small_net2_1d(width, FilterSize):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, padding="SAME"),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(4, (FilterSize,), padding="SAME"),
    )
    return Main


def small_net_1d(width, FilterSize):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, padding="SAME"),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME"),
        Relu,
        MaskedConv1d(4, (FilterSize,), padding="SAME"),
    )
    return Main
