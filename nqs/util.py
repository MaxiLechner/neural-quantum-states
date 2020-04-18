import jax.numpy as np
from jax import jit


@jit
def real_to_complex(x):
    """Turn real valued input array into complex output array."""
    return x[:, :, [0, 2]] * np.exp(x[:, :, [1, 3]] * 1j)  # shape (N,M,4) -> (N,M,2)
