from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from jax import random
import jax.numpy as np
from jax import jit
from jax.lax import fori_loop
from jax.experimental import optimizers

from network import small_net_1d
from wavefunction import log_amplitude_init
from sampler import sample_init


def initialize_heisenberg_1d(
    width, filter_size, seed, num_spins, lr, J, batch_size, pbc
):
    model = small_net_1d(width, filter_size)
    net_init, net_apply = model
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    in_shape = (-1, num_spins, 1)
    _, net_params = net_init(subkey, in_shape)
    net_apply = jit(net_apply)
    sample = sample_init(net_apply)
    logpsi = log_amplitude_init(net_apply)
    energy = energy_heisenberg_init(logpsi, J, pbc)
    grad = grad_init(logpsi)
    opt_init, opt_update, get_params = optimizers.adam(
        # optimizers.polynomial_decay(lr, 10, 0.00001, 3)
        lr
    )
    opt_state = opt_init(net_params)
    data = np.zeros((batch_size, num_spins, 1), dtype=np.float32)
    step = step_init(energy, sample, grad, logpsi, data, opt_update, get_params)
    return step, opt_state, key


def energy_heisenberg_init(log_amplitude, J, pbc):
    @jit
    def energy(net_params, state):
        @jit
        def amplitude_diff(state, i, j):
            """compute apmplitude ratio of logpsi and logpsi_fliped, where i and i+1
            have their sign fliped. As logpsi returns the real and the imaginary part
            seperately, we therefor need to recombine them into a complex valued array"""
            flip_i = np.ones(state.shape)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, j], -1)
            fliped = state * flip_i
            logpsi_fliped = log_amplitude(net_params, fliped)
            logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
            return np.exp(logpsi_fliped - logpsi)

        @jit
        def body_fun(i, loop_carry):
            E, m, s = loop_carry
            E += (
                J
                * 0.25
                * (s[:, i] * s[:, i + 1] + m[:, i] * amplitude_diff(s, i, i + 1))
            )
            return E, m, s

        def pbc_contrib(E):
            E += J * 0.25 * state[:, -1] * state[:, 0]
            E += J * 0.25 * mask[:, -1] * amplitude_diff(state, -1, 0)
            return E

        # sx*sx + sy*sy gives a contribution iff x[i]!=x[i+1]
        mask = -state * np.roll(state, -1, axis=1) + 1
        logpsi = log_amplitude(net_params, state)
        logpsi = logpsi[0] + logpsi[1] * 1j

        loop_start = 0
        loop_end = state.shape[1] - 1
        start_val = np.zeros(state.shape[0])
        start_val = start_val[..., None]
        start_val = start_val.astype("complex64")

        E, _, _ = fori_loop(loop_start, loop_end, body_fun, (start_val, mask, state))
        # Can't use if statements in jitted code, need to use lax primitive instead.
        E = jax.lax.cond(pbc, E, pbc_contrib, E, lambda E: np.add(E, 0))
        return E

    return energy


@jit
def magnetization(state):
    mag = np.sum(state, axis=1)
    return np.mean(mag)


@jit
def energy_var(energy):
    return np.var(energy)
