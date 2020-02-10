import jax
from jax import random
import jax.numpy as np
from jax import jit
from jax.lax import fori_loop
from jax.experimental import optimizers

from .network import small_net_1d, small_resnet_1d
from .wavefunction import log_amplitude_init
from .sampler import sample_init
from .optim import grad_init, step_init

import matplotlib.pyplot as plt


def initialize_model_1d(
    hamiltonian,
    width,
    filter_size,
    seed,
    num_spins,
    lr,
    J,
    batch_size,
    pbc,
    network,
    f32=True,
):
    if f32:
        f_dtype = np.float32
        c_dtype = np.complex64
    else:
        f_dtype = np.float64
        c_dtype = np.complex128

    net_dispatch = {"small_net_1d": small_net_1d, "small_resnet_1d": small_resnet_1d}
    net = net_dispatch[network]
    energy_dispatch = {
        "ising1d": energy_ising_1d_init,
        "heisenberg1d": energy_heisenberg_1d_init,
        "sutherland1d": energy_sutherland_1d_init,
    }
    energy_init = energy_dispatch[hamiltonian]

    model = net(width, filter_size)
    net_init, net_apply = model
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    in_shape = (-1, num_spins, 1)
    _, net_params = net_init(subkey, in_shape)
    net_apply = jit(net_apply)

    sample = sample_init(net_apply)
    logpsi = log_amplitude_init(net_apply)
    energy = energy_init(logpsi, J, pbc, c_dtype)

    grad = grad_init(logpsi)
    opt_init, opt_update, get_params = optimizers.adam(
        # optimizers.polynomial_decay(lr, 10, 0.00001, 3)
        lr
    )
    opt_state = opt_init(net_params)
    init_batch = np.zeros((batch_size, num_spins, 1), dtype=f_dtype)
    step = step_init(
        energy,
        sample,
        grad,
        energy_var,
        magnetization,
        logpsi,
        init_batch,
        opt_update,
        get_params,
    )
    return step, opt_state, key, get_params, net_apply, sample, logpsi, energy, grad


def energy_ising_1d_init(log_amplitude, J, pbc, c_dtype):
    @jit
    def energy(net_params, state):
        @jit
        def amplitude_diff(state, i):
            """Compute amplitude ratio of logpsi and logpsi_fliped, where spin i has its
            sign fliped. As logpsi returns the real and the imaginary part seperately,
            we therefor need to recombine them into a complex valued array"""
            flip_i = np.ones(state.shape)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
            fliped = state * flip_i
            logpsi_fliped = log_amplitude(net_params, fliped)
            logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
            return np.exp(logpsi_fliped - logpsi)  # , logpsi_fliped - logpsi
            # return logpsi_fliped - logpsi

        @jit
        def body_fun1(i, loop_carry):
            E, s = loop_carry
            E -= s[:, i] * s[:, i + 1]
            return E, s

        @jit
        def body_fun2(i, loop_carry):
            E, s = loop_carry
            E += amplitude_diff(s, i)
            return E, s

        @jit
        def body_fun(i, loop_carry):
            E, s = loop_carry
            E -= J * (amplitude_diff(s, i) + s[:, i] * s[:, i + 1])
            return E, s

        logpsi = log_amplitude(net_params, state)
        logpsi = logpsi[0] + logpsi[1] * 1j

        loop_start = 0
        loop_end = state.shape[1] - 1
        start_val = np.zeros(state.shape[0])
        start_val = start_val[..., None]
        start_val = start_val.astype(np.complex64)

        # E0, _ = fori_loop(loop_start, loop_end, body_fun1, (start_val, state))
        # diff, _ = fori_loop(loop_start, loop_end, body_fun2, (start_val, state))
        # diff += amplitude_diff(state, -1)

        E, _ = fori_loop(loop_start, loop_end, body_fun, (start_val, state))
        E -= amplitude_diff(state, -1)
        E0 = E
        diff = E

        # diff2 = np.exp(diff)
        # E = E0 + diff2
        # E = E0 - diff
        return E, E0, diff, logpsi, E, E

    return energy


def energy_heisenberg_1d_init(log_amplitude, J, pbc, c_dtype):
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

        def amp_fliped(state, i, j):
            """compute apmplitude ratio of logpsi and logpsi_fliped, where i and i+1
            have their sign fliped. As logpsi returns the real and the imaginary part
            seperately, we therefor need to recombine them into a complex valued array"""
            flip_i = np.ones(state.shape)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, j], -1)
            fliped = state * flip_i
            logpsi_fliped = log_amplitude(net_params, fliped)
            logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
            return logpsi_fliped

        @jit
        def body_fun1(i, loop_carry):
            E, s = loop_carry
            E += J * 0.25 * (s[:, i] * s[:, i + 1])
            return E, s

        @jit
        def body_fun2(i, loop_carry):
            E, m, s = loop_carry
            E += J * 0.25 * m[:, i] * amplitude_diff(s, i, i + 1)
            return E, m, s

        @jit
        def body_fun3(i, loop_carry):
            arr, s = loop_carry
            diff = amp_fliped(s, i, i + 1)
            arr = jax.ops.index_update(arr, jax.ops.index[:, i], diff)
            return arr, s

        def pbc_contrib1(E):
            E += J * 0.25 * state[:, -1] * state[:, 0]
            return E

        def pbc_contrib2(E):
            E += J * 0.25 * mask[:, -1] * amplitude_diff(state, -1, 0)
            return E

        # sx*sx + sy*sy gives a contribution iff x[i]!=x[i+1]
        mask = -state * np.roll(state, -1, axis=1) + 1
        logpsi = log_amplitude(net_params, state)
        logpsi = logpsi[0] + logpsi[1] * 1j

        logprobarr = np.zeros(state.shape, dtype=c_dtype)

        loop_start = 0
        loop_end = state.shape[1] - 1
        start_val = np.zeros(state.shape[0])
        start_val = start_val[..., None]
        start_val = start_val.astype(c_dtype)

        E0, _ = fori_loop(loop_start, loop_end, body_fun1, (start_val, state))
        E1, _, _ = fori_loop(loop_start, loop_end, body_fun2, (start_val, mask, state))
        logprobarr, _ = fori_loop(loop_start, loop_end, body_fun3, (logprobarr, state))
        # Can't use if statements in jitted code, need to use lax primitive instead.
        E0 = jax.lax.cond(pbc, E0, pbc_contrib1, E0, lambda E: np.add(E0, 0))
        E1 = jax.lax.cond(pbc, E1, pbc_contrib2, E1, lambda E: np.add(E1, 0))

        logprobarr = jax.ops.index_update(
            logprobarr, jax.ops.index[:, -1], amp_fliped(state, -1, 0)
        )

        E = E0 + E1

        return E, E0, E1, logpsi, logprobarr, mask

    return energy


def energy_sutherland_1d_init(log_amplitude, J, pbc):
    @jit
    def energy(net_params, state):
        @jit
        def amplitude_diff(state, i, j, idx):
            """compute apmplitude ratio of logpsi and logpsi_fliped, where i and i+1
            have their sign fliped. As logpsi returns the real and the imaginary part
            seperately, we therefor need to recombine them into a complex valued array"""
            state = jax.ops.index_update(state, jax.ops.index[:, i], idx[:, 0])
            state = jax.ops.index_update(state, jax.ops.index[:, j], idx[:, 1])
            logpsi_swaped = log_amplitude(net_params, state)
            logpsi_swaped = logpsi_swaped[0] + logpsi_swaped[1] * 1j
            return np.exp(logpsi_swaped - logpsi)

        @jit
        def body_fun(i, loop_carry):
            E, s = loop_carry
            idx = s[:, [i + 1, i]]
            E += amplitude_diff(state, i, i + 1, idx)
            return E, s

        def pbc_contrib(E):
            idx = state[:, [0, -1]]
            E += amplitude_diff(state, -1, 0, idx)
            return E

        logpsi = log_amplitude(net_params, state)
        logpsi = logpsi[0] + logpsi[1] * 1j

        loop_start = 0
        loop_end = state.shape[1] - 1
        start_val = np.zeros(state.shape[0])
        start_val = start_val[..., None]
        start_val = start_val.astype(np.complex64)

        E, _ = fori_loop(loop_start, loop_end, body_fun, (start_val, state))

        # Can't use if statements in jitted code, need to use lax primitive instead.
        E = jax.lax.cond(pbc, E, pbc_contrib, E, lambda E: np.add(E, 0))

        E0 = E
        E1 = E
        return E, E0, E1, logpsi

    return energy


@jit
def energy_var(energy):
    return np.var(energy)


@jit
def magnetization(state):
    mag = np.sum(state, axis=1)
    return np.mean(mag)


def callback(params, i, ax):
    E, mag, Time, epochs, gs_energy = params
    epoch_mod = 100
    if i > 0 and i % epoch_mod == 0 or i == epochs - 1:
        print(
            "{} epochs took {:.4f} seconds.".format(
                epoch_mod, Time[i] - Time[i - epoch_mod]
            )
        )
        plt.cla()
        plt.axhline(gs_energy, label="Exact Energy", color="r")
        ax.plot(E, label="Energy")
        ax.plot(mag, label="Magnetization")
        plt.legend()
        plt.draw()
        plt.pause(1.0 / 60.0)

    # visualize histogram of weights
    # pars = get_params(opt_state)
    # plt.cla()
    # _, ax = plt.subplots(1, 5, figsize=(30, 3))
    # for i in range(len(ax)):
    #     ax[i].hist(pars[i * 2][0].flatten())
    #     # ax[i].set_title("layer", i * 2)
