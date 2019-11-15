from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as np
from jax import jit, random
from jax.lax import fori_loop

from util import real_to_complex

from functools import partial


@jit
def compute_probs(arr):
    arr = real_to_complex(arr)
    tc = np.exp(arr)
    nc = np.linalg.norm(tc, 2, axis=2, keepdims=True)
    tc = tc / nc
    probs = np.square(np.abs(tc))
    return probs


@partial(jit, static_argnums=(0,))
def sample(net_apply, net_params, i, data):
    key = random.PRNGKey(i + 1)
    for i in range(data.shape[1]):
        vi = net_apply(net_params, data)
        probs = compute_probs(vi)
        key, subkey = random.split(key)
        sample = random.bernoulli(subkey, probs[:, i, 1]) * 2 - 1.0
        sample = sample.reshape(data.shape[0], 1)
        data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
    return data


@partial(jit, static_argnums=(0,))
def sample_fori(net_apply, net_params, i, data, key):
    def body(i, loop_carry):
        key, data = loop_carry
        vi = net_apply(net_params, data)
        probs = compute_probs(vi)
        key, subkey = random.split(key)
        sample = random.bernoulli(subkey, probs[:, i, 1]) * 2 - 1.0
        sample = sample.reshape(data.shape[0], 1)
        data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
        return key, data

    key, data = fori_loop(0, data.shape[1], body, (key, data))
    return key, data
