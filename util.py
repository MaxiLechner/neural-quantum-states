from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from jax.lib import pytree
from jax import jit
from jax.tree_util import tree_multimap


@jit
def real_to_complex(arr):
    _, _, s3 = arr.shape
    assert s3 == 4
    # carr = arr[:, :, [0, 2]] * np.exp(arr[:, :, [1, 3]] * 1j)
    carr = arr[:, :, [0, 2]] + arr[:, :, [1, 3]] * 1j
    return carr


@jit
def expand_dimensions(w, n):
    """expand dims of the first array by adding np.newaxis on the
    right side to match the dimension of the second array"""
    while len(w.shape) != len(n.shape):
        w = np.expand_dims(w, -1)
    return w


@jit
def apply_elementwise(eloc, jac):
    """applies the local energy in an elementwise fashion to the jacobian params,
    function modeled after tree_util functions like tree_map"""
    leaves, treedef = pytree.flatten(jac)
    out = []
    for i in leaves:
        i = expand_dimensions(eloc, i) * i
        i = i.real
        i = i.mean(axis=0)
        i = np.squeeze(i, axis=0)
        i = 2 * i
        out.append(i)
    return treedef.unflatten(out)


@jit
def make_complex(state):
    """turns the real valued state into complex form, function modeled after tree_util functions like tree_map"""
    tree_left, tree_right = state
    return tree_multimap(lambda x, y: x + y * 1j, tree_left, tree_right)
