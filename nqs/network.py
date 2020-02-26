from jax import lax
import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import (
    Relu,
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
    padding="VALID",
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
    """One-hot encodes the given indicies.
  Each index in the input ``x`` is encoded as a vector of zeros of length
  ``num_classes`` with the element at ``index`` set to one::
  >>> jax.nn.one_hot(np.array([0, 1, 2]), 3)
  DeviceArray([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)
  Indicies outside the range [0, num_classes) will be encoded as zeros::
  >>> jax.nn.one_hot(np.array([-1, 3]), 3)
  DeviceArray([[0., 0., 0.],
               [0., 0., 0.]], dtype=float32)
  Args:
    x: A tensor of indices.
    num_classes: Number of classes in the one-hot dimension.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
  """
    return np.array(x == np.arange(num_classes, dtype=x.dtype), dtype=net_dtype)


Onehot = elementwise(one_hot)


def resnet_block_1d(width, FilterSize, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, FilterSize, padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, FilterSize, padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, FilterSize, padding="SAME", net_dtype=net_dtype),
    )
    Shortcut = Identity
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum)


def small_resnet_1d(width, FilterSize, one_hot=True, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, padding="SAME", net_dtype=net_dtype),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        Relu,
        resnet_block_1d(width, (FilterSize,), net_dtype=net_dtype),
        Relu,
        MaskedConv1d(4, (FilterSize,), padding="SAME", net_dtype=net_dtype),
    )
    if one_hot:
        return stax.serial(Onehot, Main)
    else:
        return Main


def small_net2_1d(width, FilterSize, one_hot=True, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(4, (FilterSize,), padding="SAME", net_dtype=net_dtype),
    )
    if one_hot:
        return stax.serial(Onehot, Main)
    else:
        return Main


def small_net_1d(width, FilterSize, one_hot=True, net_dtype=np.float32):
    Main = stax.serial(
        MaskedConv1d(width, (FilterSize,), True, padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(width, (FilterSize,), padding="SAME", net_dtype=net_dtype),
        Relu,
        MaskedConv1d(4, (FilterSize,), padding="SAME", net_dtype=net_dtype),
    )
    if one_hot:
        return stax.serial(Onehot, Main)
    else:
        return Main
