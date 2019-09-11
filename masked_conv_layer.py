# import numpy as onp
# from jax import grad, jit
from jax import lax

# from jax.experimental import stax
# from jax import random
# from jax.experimental.stax import GeneralConv, Relu, relu, _elemwise_no_params
# import jax
import jax.numpy as np

# import matplotlib as mpl
# import functools

# import pdb

from jax.experimental.stax import randn, glorot
import itertools


def MaskedConv1d(
    out_chan,
    filter_shape,
    num_layer=1,
    strides=None,
    padding="VALID",
    W_init=None,
    b_init=randn(1e-6),
):
    """Layer construction function for a general convolution layer."""
    assert len(filter_shape) == 1, "filter_shape must be one dimensional"
    assert filter_shape[0] % 2 != 0, "filter_shape must be odd"

    def computefilters(num_layer):
        if num_layer == 0:
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

    filters = computefilters(num_layer)
    # print(filters)
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
        W, b = W_init(rng, kernel_shape), b_init(rng, bias_shape)
        return output_shape, (W * filters, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return (
            lax.conv_general_dilated(
                inputs, W, strides, padding, one, one, dimension_numbers
            )
            + b
        )

    return init_fun, apply_fun