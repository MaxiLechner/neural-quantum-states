import jax
import jax.numpy as jnp
import flax
from flax import nn
from flax.nn.linear import MaskedConv1d


def one_hot(x, num_classes=2, net_dtype=jnp.float32):
    """One-hot encodes the given indicies."""
    return jnp.array(x == jnp.arange(num_classes, dtype=x.dtype), dtype=net_dtype)


def real_to_complex(x):
    """Turn real valued input array into complex valued output array."""
    return x[:, :, [0, 2]] * jnp.exp(x[:, :, [1, 3]] * 1j)  # shape (N,M,4) -> (N,M,2)


class LSTM(nn.Module):
    """LSTM encoder. Turns a sequence of vectors into a vector."""

    def apply(self, inputs, key, hidden_size=24):
        # inputs.shape = <float32>[batch_size, seq_length, emb_size].
        batch_size = inputs.shape[0]
        carry = nn.LSTMCell.initialize_carry(key, (batch_size,), hidden_size)
        _, outputs = flax.jax_utils.scan_in_dim(
            nn.LSTMCell.partial(name="lstm_cell"), carry, inputs, axis=1
        )
        return outputs


@nn.module
def lstm(x, key, depth, hidden_size=24, use_one_hot=True, dtype=jnp.float32):
    keys = jax.random.split(key, depth)
    if use_one_hot:
        x = one_hot(x)
    for i in range(depth):
        x = LSTM(x, keys[i], hidden_size=hidden_size)
        x = nn.relu(x)
    # x = LSTM(x, keys[0], hidden_size=hidden_size)
    # x = nn.relu(x)
    # x = LSTM(x, keys[1], hidden_size=hidden_size)
    # x = nn.relu(x)
    # x = LSTM(x, keys[2], hidden_size=hidden_size)
    # x = nn.relu(x)
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
