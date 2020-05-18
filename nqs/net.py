import jax.numpy as jnp
from jax import lax, random

from flax import nn, jax_utils
from flax.nn import base, initializers

import numpy as onp

default_kernel_init = initializers.lecun_normal()


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def one_hot(x, num_classes=2, net_dtype=jnp.float32):
    """One-hot encodes the given indicies."""
    return jnp.array(x == jnp.arange(num_classes, dtype=x.dtype), dtype=net_dtype)


def real_to_complex(x):
    """Turn real valued input array into complex valued output array."""
    return x[:, :, [0, 2]] * jnp.exp(x[:, :, [1, 3]] * 1j)  # shape (N,M,4) -> (N,M,2)


class MaskedConv1d(base.Module):
    """Masked 1D Convolution Module based on flax.nn.linear.Conv ."""

    def apply(
        self,
        inputs,
        features,
        kernel_size,
        is_first_layer=False,
        strides=None,
        padding="SAME",
        input_dilation=None,
        kernel_dilation=None,
        feature_group_count=1,
        bias=True,
        dtype=jnp.float32,
        precision=None,
        kernel_init=default_kernel_init,
        bias_init=initializers.zeros,
    ):
        """Applies a convolution to the inputs.

    """

        inputs = jnp.asarray(inputs, dtype)

        assert len(kernel_size) == 1, "kernel_shape must be one dimensional"
        assert kernel_size[0] % 2 != 0, "kernel_shape must be odd"

        mask = onp.ones(kernel_size[0])
        if is_first_layer:
            i = (kernel_size[0] - 1) // 2
        else:
            i = (kernel_size[0] + 1) // 2
        mask[i:] = 0
        mask = jnp.asarray(mask[:, onp.newaxis, onp.newaxis], dtype)

        if strides is None:
            strides = (1,) * (inputs.ndim - 2)

        in_features = inputs.shape[-1]
        assert in_features % feature_group_count == 0
        kernel_shape = kernel_size + (in_features // feature_group_count, features)
        kernel = self.param("kernel", kernel_shape, kernel_init)
        kernel = jnp.asarray(kernel, dtype)
        kernel = kernel * mask

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        y = lax.conv_general_dilated(
            inputs,
            kernel,
            strides,
            padding,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
            precision=precision,
        )

        if bias:
            bias = self.param("bias", (features,), bias_init)
            bias = jnp.asarray(bias, dtype)
            y = y + bias
        return y


class LSTM(nn.Module):
    """LSTM encoder. Turns a sequence of vectors into a vector."""

    def apply(self, inputs, key, hidden_size=24):
        # inputs.shape = <float32>[batch_size, seq_length, emb_size].
        batch_size = inputs.shape[0]
        carry = nn.LSTMCell.initialize_carry(key, (batch_size,), hidden_size)
        _, outputs = jax_utils.scan_in_dim(
            nn.LSTMCell.partial(name="lstm_cell"), carry, inputs, axis=1
        )
        return outputs


@nn.module
def lstm(x, key, depth, hidden_size=24, use_one_hot=True, dtype=jnp.float32):
    keys = random.split(key, depth)
    if use_one_hot:
        x = one_hot(x)
    for i in range(depth):
        x = LSTM(x, keys[i], hidden_size=hidden_size)
        x = nn.relu(x)
    x = nn.Dense(x, 4)
    x = real_to_complex(x)
    return x


@nn.module
def conv(x, depth, features, kernel_size, use_one_hot=True, dtype=jnp.float32):
    if use_one_hot:
        x = one_hot(x)
    x = MaskedConv1d(x, features, (kernel_size,), dtype=dtype, is_first_layer=True)
    x = nn.relu(x)
    for _ in range(depth - 2):
        x = MaskedConv1d(x, features, (kernel_size,), dtype=dtype)
        x = nn.relu(x)
    x = MaskedConv1d(x, 4, (kernel_size,), dtype=dtype)
    x = real_to_complex(x)
    return x
