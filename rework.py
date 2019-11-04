#!/usr/bin/env python
# coding: utf-8

import jax.numpy as np
import jax

# from wavefunction import psi, sample, logpsi
from network import net

from jax import random

# from jax import jit, grad
# from jax.experimental import optimizers
# from matplotlib import pyplot as plt
# from jax import vmap

from functools import partial

# import pdb


def real_to_complex(arr):
    _, _, s3 = arr.shape
    assert s3 == 4
    carr = arr[:, :, [0, 2]] * np.exp(arr[:, :, [1, 3]] * 1j)
    return carr


def compute_probs(arr):
    arr = real_to_complex(arr)
    tc = np.exp(arr)
    nc = np.linalg.norm(tc, 2, axis=2, keepdims=True)
    tc = tc / nc
    probs = np.square(np.abs(tc))
    return probs


def lpsi(net, data):
    """compute logpsi for a batch of samples. As the network returns
    the amplitude for both up and down states we need to pick the
    right amplitude by indexing according to the samples"""
    arr = net(data)
    arr = real_to_complex(arr)
    tc = np.exp(arr)
    nc = np.linalg.norm(tc, 2, axis=2, keepdims=True) ** 2

    idx = (data + 1) / 2
    idx = idx.astype(int)
    B, N, M = data.shape
    splits = np.split(arr, B)
    splits = [i.reshape(N, 2) for i in splits]
    isplits = np.split(idx, B)
    isplits = [i.reshape(N) for i in isplits]
    vi = np.stack([splits[j][np.arange(N), isplits[j]] for j in range(B)]).reshape(
        B, N, 1
    )

    logpsi = vi - 0.5 * np.log(nc)
    logpsi = np.sum(logpsi, axis=1)
    return np.real(logpsi), np.imag(logpsi)


def sample_one_batch(net, key, data):
    for i in range(data.shape[1]):
        vi = net(data)
        probs = compute_probs(vi)
        key, subkey = random.split(key)
        sample = random.bernoulli(subkey, probs[:, i, 0]) * 2 - 1.0
        sample = sample.reshape(data.shape[0], 1)
        data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
    return data


from jax.lib import pytree


def make_complex(state):
    """turns the real valued state into complex form, function modeled after tree_util functions like tree_map"""
    a, b = state
    assert len(a) == len(b)
    leaves, treedef = pytree.flatten(a)
    leaves2, _ = pytree.flatten(b)
    out = []
    for i in range(len(leaves)):
        out.append(leaves[i] + leaves2[i] * 1j)
    return treedef.unflatten(out)


def Eloc(lpsi, state):
    def amplitude_diff(state, i):
        """logpsi returns the real and the imaginary part seperately,
        we therefor need to recombine them into a complex valued array"""
        logpsi = lpsi(state)
        logpsi = logpsi[0] + logpsi[1] * 1j
        flip_i = np.ones(state.shape)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
        fliped = state * flip_i
        logpsi_fliped = lpsi(fliped)
        logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
        return np.exp(logpsi_fliped - logpsi)

    E = 0
    for i in range(state.shape[1] - 1):
        E += (state[:, i] * state[:, i + 1] + 1) * amplitude_diff(state, i)
    return E


"""define network and input parameters"""
N = 5  # input size
M = (
    2 * 2
)  # number of possible values each input can take times 2 as lax.conv only works with real valued weights
B = 2  # batchsize
# data = np.zeros((B, N, 1), dtype=np.float32)
FilterSize = 5  # width of conv, (FilterSize-1)/2 nonzero elements, must be odd
model = net(M, FilterSize)
########################################################################################
net_init, net_apply = model
key = random.PRNGKey(1)
key, subkey = random.split(key)
in_shape = (1, N, 1)
out_shape, net_params = net_init(subkey, in_shape)
data = random.bernoulli(key, 0.2, (B, N, 1)) * 2 - 1.0

net = partial(net_apply, net_params)
sob = sample_one_batch(net, key, data)

# print(lpsi(sob))

# eloc = Eloc(lpsi, sob)
# mean = np.mean(eloc.conj())
# print(eloc)
# print(mean)
# print(eloc - mean)
