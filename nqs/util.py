import jax.numpy as np
from jax import jit
from jax.tree_util import tree_map, tree_multimap


@jit
def real_to_complex(arr):
    _, _, s3 = arr.shape
    assert s3 == 4
    carr = arr[:, :, [0, 2]] * np.exp(arr[:, :, [1, 3]] * 1j)
    return carr


@jit
def expand_dimensions(w, n):
    """Expand dims of the first array by adding np.newaxis on the
    right side to match the dimension of the second array."""
    while len(w.shape) != len(n.shape):
        w = np.expand_dims(w, -1)
    return w


@jit
def apply_elementwise(eloc, jac):
    """Applies the local energy in an elementwise fashion to the jacobian params."""

    def body(i):
        i = expand_dimensions(eloc, i) * i
        i = i.real
        i = i.mean(axis=0)
        i = np.squeeze(i, axis=0)
        i = 2 * i
        return i

    return tree_map(body, jac)


@jit
def make_complex(state):
    """Recombines the real and imaginary part."""
    tree_left, tree_right = state
    return tree_multimap(lambda x, y: x + y * 1j, tree_left, tree_right)
