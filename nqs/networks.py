import jax
import jax.numpy as jnp
from jax import lax, random, jit

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
    x = (x + 1) / 2
    return jnp.array(x == jnp.arange(num_classes, dtype=x.dtype), dtype=net_dtype)


# def real_to_complex(x):
#     """Turn real valued input array into complex valued output array."""
#     return x[..., [0, 2]] * jnp.exp(x[..., [1, 3]] * 1j)  # shape (N,M,4) -> (N,M,2)


@jit
def prob(x):
    x = jnp.exp(x)
    norm = jnp.linalg.norm(x, 2, axis=-1, keepdims=True)
    x = x / norm
    probs = jnp.square(jnp.abs(x))
    return probs


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


class real_to_complex(nn.Module):
    """Turn real valued input array into complex valued output array."""

    def apply(self, x):
        real = nn.Dense(x, 10)
        real = nn.relu(real)
        real = nn.Dense(real, 2)

        imag = nn.Dense(x, 10)
        imag = nn.relu(imag)
        imag = nn.Dense(imag, 2)
        imag = jnp.pi * nn.soft_sign(imag)
        return real * jnp.exp(1j * imag)


class MultiLSTMCell(nn.Module):
    """LSTM encoder. Turns a sequence of vectors into a vector."""

    def apply(self, carry, x, depth, use_one_hot):
        carry_list = []
        if use_one_hot:
            x = one_hot(x)
        for i in range(depth):
            c, x = nn.LSTMCell(carry[i], x)  # pylint: disable=unpacking-non-sequence
            x = nn.relu(x)
            carry_list.append(c)
        # x = nn.Dense(x, 4)
        x = real_to_complex(x)
        return carry_list, x


class lstm(nn.Module):
    def apply(self, x, init_config, depth, hidden_size, use_one_hot):
        carry, cell = self._init(init_config, depth, hidden_size, use_one_hot)
        _, x = jax_utils.scan_in_dim(cell, carry, x, axis=1)
        return x

    def _init(self, init_config, depth, hidden_size, use_one_hot):
        batch_size = init_config.shape[0]
        # use dummy key as lstm is always initialized with all zeros
        init_key = random.PRNGKey(0)
        carry = nn.LSTMCell.initialize_carry(init_key, (batch_size,), hidden_size)
        carry_list = [carry for i in range(depth)]
        cell = MultiLSTMCell.partial(depth=depth, use_one_hot=use_one_hot)
        return carry_list, cell

    @nn.module_method
    def sample(self, init_config, key, depth, hidden_size, use_one_hot):
        carry, cell = self._init(init_config, depth, hidden_size, use_one_hot)

        @jit
        def body(i, loop_carry):
            key, carry, config = loop_carry
            carry, out = cell(carry, config[:, i, :])
            probs = prob(out)
            key, subkey = random.split(key)
            sample = random.bernoulli(subkey, probs[..., 1]) * 2 - 1.0
            sample = sample[..., jnp.newaxis]
            config = jax.ops.index_update(config, jax.ops.index[:, i + 1], sample)
            return key, carry, config

        # need to increase dimension of init_config by one in order to sample a state of
        # length N as config[:,i+1] needs to depend on config[:,i].
        a, _, c = init_config.shape
        w = jnp.zeros((a, 1, c))
        init_config = jnp.hstack([w, init_config])
        key, _, config = lax.fori_loop(
            0, init_config.shape[1], body, (key, carry, init_config)
        )
        config = config[:, 1:, :]
        return key, config


class Conv(nn.Module):
    def apply(self, x, depth, features, kernel_size, use_one_hot):
        if use_one_hot:
            x = one_hot(x)
        x = MaskedConv1d(x, features, (kernel_size,), is_first_layer=True)
        x = nn.relu(x)
        for _ in range(depth - 2):
            x = MaskedConv1d(x, features, (kernel_size,))
            x = nn.relu(x)
        x = MaskedConv1d(x, 4, (kernel_size,))
        x = real_to_complex(x)
        return x


class conv(nn.Module):
    def apply(self, x, init_config, depth, features, kernel_size, use_one_hot):
        conv = self._init(depth, features, kernel_size, use_one_hot)
        return conv(x)

    def _init(self, depth, features, kernel_size, use_one_hot):
        return Conv.partial(
            depth=depth,
            features=features,
            kernel_size=kernel_size,
            use_one_hot=use_one_hot,
        )

    @nn.module_method
    def sample(self, key, init_config, depth, features, kernel_size, use_one_hot):
        model = self._init(depth, features, kernel_size, use_one_hot)

        @jit
        def body(i, loop_carry):
            key, config = loop_carry
            out = model(config)
            probs = prob(out)
            key, subkey = random.split(key)
            sample = random.bernoulli(subkey, probs[:, i, 1]) * 2 - 1.0
            sample = sample[..., jnp.newaxis]
            config = jax.ops.index_update(config, jax.ops.index[:, i], sample)
            return key, config

        key, config = lax.fori_loop(0, init_config.shape[1], body, (key, init_config))
        return key, config
