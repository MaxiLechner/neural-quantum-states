import jax
from jax import random
import jax.numpy as np
from network import net
from wavefunction import lpsi, make_complex, compute_probs, apply_elementwise
from jax import jit, vmap

from jax.experimental import optimizers
from time import time

import matplotlib.pyplot as plt

from functools import partial
import pdb

# from jax.lax import fori_loop


def model2(net_apply, net_params, lpsi, Eloc):
    def sample(key):
        data = np.zeros((B, N, 1), dtype=np.float32)
        for i in range(data.shape[1]):
            vi = net_apply(net_params, data)
            probs = compute_probs(vi)
            key, subkey = random.split(key)
            sample = random.bernoulli(subkey, probs[:, i, 0]) * 2 - 1.0
            sample = sample.reshape(data.shape[0], 1)
            data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
        return data

    def energy(state):
        def amplitude_diff(i):
            """logpsi returns the real and the imaginary part seperately,
            we therefor need to recombine them into a complex valued array"""
            flip_i = np.ones(state.shape)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
            fliped = state * flip_i
            logpsi_fliped = lpsi(net_apply, net_params, fliped)
            logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
            return np.exp(logpsi_fliped - logpsi)

        logpsi = lpsi(net_apply, net_params, state)
        logpsi = logpsi[0] + logpsi[1] * 1j
        E = 0
        for i in range(state.shape[1] - 1):
            E -= (state[:, i] * state[:, i + 1]) - amplitude_diff(i)
        E -= amplitude_diff(-1)
        return E

    # @partial(jit, static_argnums=(0, 3))
    def grad(state, energy):
        eloc = energy.conj()
        eloc_mean = np.mean(eloc)
        eloc = eloc - eloc_mean
        jac = jax.jacrev(lpsi, argnums=1)
        jac = jac(net_apply, net_params, state)
        jac = make_complex(jac)
        jac = apply_elementwise(eloc, jac)
        return jac

    return sample, energy, grad


@partial(jit, static_argnums=(0,))
def sample(net_apply, net_params, key):
    data = np.zeros((B, N, 1), dtype=np.float32)
    for i in range(data.shape[1]):
        vi = net_apply(net_params, data)
        probs = compute_probs(vi)
        key, subkey = random.split(key)
        sample = random.bernoulli(subkey, probs[:, i, 0]) * 2 - 1.0
        sample = sample.reshape(data.shape[0], 1)
        data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
    return key, data


@partial(jit, static_argnums=(0, 3))
def energy(net_apply, net_params, state, lpsi):
    @jit
    def amplitude_diff(state, i):
        """logpsi returns the real and the imaginary part seperately,
        we therefor need to recombine them into a complex valued array"""
        flip_i = np.ones(state.shape)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
        fliped = state * flip_i
        logpsi_fliped = lpsi(net_apply, net_params, fliped)
        logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
        return np.exp(logpsi_fliped - logpsi)

    logpsi = lpsi(net_apply, net_params, state)
    logpsi = logpsi[0] + logpsi[1] * 1j
    E = 0
    for i in range(state.shape[1] - 1):
        E -= state[:, i] * state[:, i + 1] - amplitude_diff(state, i)
    E -= amplitude_diff(state, -1)
    # print(E)
    return E


# @partial(jit, static_argnums=(0, 3))
def energy2(net_apply, net_params, state, lpsi):
    # @jit
    def amplitude_diff(state, i):
        """logpsi returns the real and the imaginary part seperately,
        we therefor need to recombine them into a complex valued array"""
        flip_i = np.ones(state.shape)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
        fliped = state * flip_i
        logpsi_fliped = lpsi(net_apply, net_params, fliped)
        logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
        print("-" * 50)
        print(np.exp(logpsi_fliped - logpsi))
        print(np.exp(logpsi_fliped - logpsi).mean())
        print("=" * 50)
        return np.exp(logpsi_fliped - logpsi)

    logpsi = lpsi(net_apply, net_params, state)
    logpsi = logpsi[0] + logpsi[1] * 1j
    E = 0
    for i in range(state.shape[1] - 1):
        E -= state[:, i] * state[:, i + 1] - amplitude_diff(state, i)
    E -= amplitude_diff(state, -1)
    # print(E)
    return E


@partial(jit, static_argnums=(0, 3))
def grad(net_apply, net_params, state, lpsi, energy):
    eloc = energy.conj()
    eloc_mean = np.mean(eloc)
    eloc = eloc - eloc_mean
    jac = jax.jacrev(lpsi, argnums=1)
    jac = jac(net_apply, net_params, state)
    jac = make_complex(jac)
    jac = apply_elementwise(eloc, jac)
    return jac


@jit
def magnetization(state):
    E = np.sum(state, axis=1)
    return E


def callback(params, i):
    E, mag, end_time, start_time = params
    print("iteration {} took {:.4f} secs.".format(i + 1, end_time - start_time))
    plt.cla()
    ax.plot(E, label="Energy")
    ax.plot(mag, label="Magnetization")
    plt.draw()
    plt.pause(1.0 / 60.0)


# @partial(jit, static_argnums=(0,))
def step(i, key, opt_state):
    params = get_params(opt_state)
    key, s = sample(net_apply, params, key)
    e = energy(net_apply, params, s, lpsi)
    g = grad(net_apply, params, s, lpsi, e)
    E.append(e.real.mean())
    mag.append(magnetization(s).mean())
    return opt_update(i, g, opt_state)


if __name__ == "__main__":
    N = 10
    M = (
        2 * 2
    )  # number of possible values each input can take times 2 as lax.conv only works with real valued weights
    B = 100  # batchsize
    FilterSize = 3
    # parameters = (N, M, B, FilterSize)
    model = net(M, FilterSize)
    ########################################################################################
    net_init, net_apply = model
    key = random.PRNGKey(1)
    key, subkey = random.split(key)
    in_shape = (-1, N, 1)
    out_shape, net_params = net_init(subkey, in_shape)
    net_apply = jit(net_apply)
    # sample, energy, grad = model(parameters, net, lpsi)
    # sample = sample(net_apply, net_params, key)
    # energy = energy(net_apply, net_params, sample, lpsi)
    # grad = grad(net_apply, net_params, sample, lpsi, energy)
    # print(sample)
    # print(energy)
    # print(grad[0][0][1])

    opt_init, opt_update, get_params = optimizers.adam(1e-02)

    E = []
    mag = []
    ratio = []

    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)

    opt_state = opt_init(net_params)
    for i in range(35):
        start_time = time()
        opt_state = step(i, key, opt_state)
        end_time = time()
        callback((E, mag, end_time, start_time), i)

    net_params = get_params(opt_state)
    plt.show(block=True)
    # print(net_params)
    # gs_energy = 1 - 1 / (np.sin(np.pi / (2 * (2 * N + 1))))
    # print("Energy: ", np.mean(np.array(E)[750:]))
    # print("Magnetization: ", np.mean(np.array(mag)[750:]))
    # print(gs_energy)

    for i in range(1):
        key, s = sample(net_apply, net_params, key)
        print(s)
        e = energy(net_apply, net_params, s, lpsi)
        print(e)
        print(np.mean(e))
        print("=" * 100)
    pdb.set_trace()

    # plt.plot(E)
    # plt.plot(mag)
    # plt.show()
    # pdb.set_trace()
    # arr1 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).reshape(
    #     1, 10, 1
    # )
    # arr2 = np.ones((1, 10, 1))
    # print(energy(net_apply, net_params, arr1, lpsi))
    # print(energy(net_apply, net_params, arr2, lpsi))
