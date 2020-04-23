from jax import lax
import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import (
    Relu,
    BatchNorm,
    FanOut,
    FanInSum,
    Identity,
    randn,
    glorot,
    elementwise,
)

import itertools


def MaskedConv1d(
    out_chan,
    filter_shape,
    is_first_layer=False,
    strides=None,
    padding="SAME",
    W_init=None,
    b_init=randn(1e-6),
    net_dtype=np.float32,
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
            W_init(rng, kernel_shape, dtype=net_dtype),
            b_init(rng, bias_shape, dtype=net_dtype),
        )
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        W = W * mask
        return (
            lax.conv_general_dilated(
                inputs, W, strides, padding, one, one, dimension_numbers
            )
            + b
        )

    return init_fun, apply_fun


def one_hot(x, num_classes=2, net_dtype=np.float32):
    """One-hot encodes the given indicies."""
    return np.array(x == np.arange(num_classes, dtype=x.dtype), dtype=net_dtype)


def real_to_complex(x):
    """Turn real valued input array into complex valued output array."""
    return x[:, :, [0, 2]] * np.exp(x[:, :, [1, 3]] * 1j)  # shape (N,M,4) -> (N,M,2)


Onehot = elementwise(one_hot)
Real_to_complex = elementwise(real_to_complex)


def resnet_block_1d(width, FilterSize, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, FilterSize, net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, FilterSize, net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, FilterSize, net_dtype=net_dtype),
    )
    Shortcut = Identity
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum)


def small_resnet_1d(width, FilterSize, one_hot=True, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, net_dtype=net_dtype),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        Relu,
        MaskedConv1d(4, (FilterSize,), net_dtype=net_dtype),
        Real_to_complex,
    )
    if one_hot:
        return stax.serial(Onehot, Main)
    else:
        return Main


def small_resnetbn_1d(width, FilterSize, axis, one_hot=True, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, net_dtype=net_dtype),
        BatchNorm(axis=axis),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        BatchNorm(axis=axis),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        BatchNorm(axis=axis),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        BatchNorm(axis=axis),
        Relu,
        MaskedConv1d(4, (FilterSize,), net_dtype=net_dtype),
        Real_to_complex,
    )
    if one_hot:
        return stax.serial(Onehot, Main)
    else:
        return Main


def small_net_1d(width, FilterSize, one_hot=True, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, (FilterSize,), net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, (FilterSize,), net_dtype=net_dtype),
        Relu,
        MaskedConv1d(4, (FilterSize,), net_dtype=net_dtype),
        Real_to_complex,
    )
    if one_hot:
        return stax.serial(Onehot, Main)
    else:
        return Main


def small_netbn_1d(width, FilterSize, axis, one_hot=True, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, net_dtype=net_dtype),
        BatchNorm(axis=axis),
        Relu,
        MaskedConv1d(width, (FilterSize,), net_dtype=net_dtype),
        BatchNorm(axis=axis),
        Relu,
        MaskedConv1d(width, (FilterSize,), net_dtype=net_dtype),
        BatchNorm(axis=axis),
        Relu,
        MaskedConv1d(4, (FilterSize,), net_dtype=net_dtype),
        Real_to_complex,
    )
    if one_hot:
        return stax.serial(Onehot, Main)
    else:
        return Main
