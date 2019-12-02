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


def sample_init(net_apply):
    @jit
    def sample(net_params, data, key):
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

    return sample


# # @partial(jit, static_argnums=(0,))
# def check_sample(net_apply, net_params, data, key):
#     def body(i, loop_carry):
#         key, data, _ = loop_carry
#         vi = net_apply(net_params, data)
#         probs = compute_probs(vi)
#         key, subkey = random.split(key)
#         sample = random.bernoulli(subkey, probs[:, i, 1]) * 2 - 1.0
#         sample = sample.reshape(data.shape[0], 1)
#         data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
#         return key, data, probs

#     # pdb.set_trace()
#     stuff = np.ones((100, 10, 2))
#     # assert np.allclose(np.sum(probs, axis=2), 1)
#     print("=" * 100)
#     # print(data[0])
#     # print("-" * 100)
#     key, data, probs = fori_loop(0, data.shape[1], body, (key, data, stuff))
#     print(data[0])
#     print("_" * 100)
#     print(probs[0])
#     assert np.allclose(np.sum(probs, axis=2), 1)
#     return key, data


# @partial(jit, static_argnums=(0,))
def check_sample(net_apply, net_params, data, key):
    for i in range(data.shape[1]):
        vi = net_apply(net_params, data)
        probs = compute_probs(vi)
        key, subkey = random.split(key)
        sample = random.bernoulli(subkey, probs[:, i, 1]) * 2 - 1.0
        print("{}\t{}\t{}".format(i, vi[0, i], probs[0, i]))
        # print("{}\t{}".format(i, probs[:, i, 1]))
        print("{}\t{}".format(i, sample))
        sample = sample.reshape(data.shape[0], 1)
        data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
    print("=" * 100)
    # pdb.set_trace()
    # assert np.allclose(np.sum(probs, axis=2), 1)
    # print("=" * 100)
    # print(data[0])
    # print("-" * 100)
    # key, data, probs = fori_loop(0, data.shape[1], body, (key, data, stuff))
    # print(data[0])
    # print("_" * 100)
    # print(probs[0])
    # assert np.allclose(np.sum(probs, axis=2), 1)
    return key, data
