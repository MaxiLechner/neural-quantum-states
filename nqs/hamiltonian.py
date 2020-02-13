import jax
from jax import random
import jax.numpy as np
from jax import jit, vmap
from jax.lax import fori_loop
from jax.experimental import optimizers

from .network import small_net_1d, small_resnet_1d
from .wavefunction import log_amplitude_init
from .sampler import sample_init
from .optim import grad_init, step_init
from .util import real_to_complex

import matplotlib.pyplot as plt
from functools import partial


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

    net_dtype = np.float32

    net_dispatch = {"small_net_1d": small_net_1d, "small_resnet_1d": small_resnet_1d}

    energy_dispatch = {
        "ising1d": energy_ising_1d_init,
        "heisenberg1d": energy_heisenberg_1d_init,
        "sutherland1d": energy_sutherland_1d_init,
    }

    try:
        net = net_dispatch[network]
    except KeyError:
        print(
            f"{network} is not a valid network. You can choose between small_net_1d and small_resnet_1d."
        )
        raise

    try:
        energy_init = energy_dispatch[hamiltonian]
    except KeyError:
        print(f"{hamiltonian} is not a valid hamiltonian. You can choose between ising1d and heisenberg1d.")
        raise

    model = net(width, filter_size, net_dtype=net_dtype)
    net_init, net_apply = model
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    in_shape = (-1, num_spins, 1)
    _, net_params = net_init(subkey, in_shape)
    net_apply = jit(net_apply)

    sample = sample_init(net_apply)
    logpsi = log_amplitude_init(net_apply)
    energy = energy_init(logpsi, net_apply, J, pbc, c_dtype)

    grad = grad_init(logpsi)
    opt_init, opt_update, get_params = optimizers.adam(lr)
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


def energy_ising_1d_init(log_amplitude, net_apply, J, pbc, c_dtype):
    @jit
    def energy(net_params, config):
        @jit
        def amplitude_diff(config, i):
            """Compute amplitude ratio of logpsi and logpsi_flipped, where spin i has its
            sign flipped. As logpsi returns the real and the imaginary part seperately,
            we therefor need to recombine them into a complex valued array"""
            flip_i = np.ones(config.shape)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
            flipped = config * flip_i
            logpsi_flipped = log_amplitude(net_params, flipped)
            logpsi_flipped = logpsi_flipped[0] + logpsi_flipped[1] * 1j
            return np.exp(logpsi_flipped - logpsi)

        @jit
        def body_fun(i, loop_carry):
            E, s = loop_carry
            E -= J * (amplitude_diff(s, i) + s[:, i] * s[:, i + 1])
            return E, s

        def pbc_contrib(E):
            E -= J * config[:, -1] * config[:, 0]
            return E

        logpsi = log_amplitude(net_params, config)
        logpsi = logpsi[0] + logpsi[1] * 1j
        logpsi = logpsi.astype(c_dtype)

        start = 0
        end = config.shape[1] - 1
        start_val = np.zeros(config.shape[0], dtype=c_dtype)[..., None]

        E, _ = fori_loop(start, end, body_fun, (start_val, config))
        E -= amplitude_diff(config, -1)

        # Can't use if statements in jitted code, need to use lax primitive instead.
        E = jax.lax.cond(pbc, E, pbc_contrib, E, lambda x: x)

        return E, E, E, logpsi, E, E, E, E

    return energy


def energy_heisenberg_1d_init(log_amplitude, net_apply, J, pbc, c_dtype):
    @jit
    def energy(net_params, config):
        @jit
        def amplitude_diff(config, i, j):
            """compute apmplitude ratio of logpsi and logpsi_flipped, where i and i+1
            have their sign flipped. As logpsi returns the real and the imaginary part
            seperately, we therefor need to recombine them into a complex valued array"""
            flip_i = np.ones(config.shape)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, j], -1)
            flipped = config * flip_i
            logpsi_flipped = log_amplitude(net_params, flipped)
            logpsi_flipped = logpsi_flipped[0] + logpsi_flipped[1] * 1j
            return np.exp(logpsi_flipped - logpsi)

        @jit
        def amp_fliped(config, i, j):
            """compute apmplitude ratio of logpsi and logpsi_flipped, where i and i+1
            have their sign flipped. As logpsi returns the real and the imaginary part
            seperately, we therefor need to recombine them into a complex valued array"""
            flip_i = np.ones(config.shape)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, j], -1)
            flipped = config * flip_i
            logpsi_flipped = log_amplitude(net_params, flipped)
            logpsi_flipped = logpsi_flipped[0] + logpsi_flipped[1] * 1j
            return logpsi_flipped

        @jit
        def _vi_fliped(config, i, j):
            """compute apmplitude ratio of logpsi and logpsi_flipped, where i and i+1
            have their sign flipped. As logpsi returns the real and the imaginary part
            seperately, we therefor need to recombine them into a complex valued array"""

            def index(x, y, i):
                xi = x[i]  # shape: (N,2)
                yi = y[i]  # shape: (N)
                arange = np.arange(xi.shape[0])
                return xi[arange, yi]

            flip_i = np.ones(config.shape)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
            flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, j], -1)
            flipped = config * flip_i
            vi_fliped = net_apply(net_params, flipped)
            vi_fliped = real_to_complex(vi_fliped)

            B, _, _ = config.shape
            idx = (config + 1) / 2
            idx = idx.astype(np.int32).squeeze()
            index = vmap(partial(index, vi_fliped, idx))
            vi_fliped = index(np.arange(B))[..., np.newaxis]
            # vi_fliped = np.sum(vi_fliped, axis=1)
            return vi_fliped

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
            # print("arr: ", arr.shape)
            # print("diff: ", diff.shape)
            arr = jax.ops.index_update(arr, jax.ops.index[:, i], diff)
            return arr, s

        @jit
        def body_fun4(i, loop_carry):
            arr, s = loop_carry
            vi = _vi_fliped(s, i, i + 1)
            # print("arr: ", arr.shape)
            # print("vi: ", vi.shape)
            arr = jax.ops.index_update(arr, jax.ops.index[:, i], vi)
            return arr, s

        def pbc_contrib1(E):
            E += J * 0.25 * config[:, -1] * config[:, 0]
            return E

        def pbc_contrib2(E):
            E += J * 0.25 * mask[:, -1] * amplitude_diff(config, -1, 0)
            return E

        def index(x, y, i):
            xi = x[i]  # shape: (N,2)
            yi = y[i]  # shape: (N)
            arange = np.arange(xi.shape[0])
            return xi[arange, yi]

        # sx*sx + sy*sy gives a contribution iff x[i]!=x[i+1]
        mask = config * np.roll(config, -1, axis=1) - 1
        logpsi = log_amplitude(net_params, config)
        logpsi = logpsi[0] + logpsi[1] * 1j
        logpsi = logpsi.astype(c_dtype)

        vi = net_apply(net_params, config)
        vi = real_to_complex(vi)

        B, N, _ = config.shape
        logprobarr = np.zeros(config.shape, dtype=c_dtype)
        vi_fliped = np.zeros((B, N, N, 1), dtype=c_dtype)

        idx = (config + 1) / 2
        idx = idx.astype(np.int32).squeeze()
        index = vmap(partial(index, vi, idx))
        vi = index(np.arange(B))[..., np.newaxis]
        # vi = np.sum(vi, axis=1)

        start = 0
        end = config.shape[1] - 1
        start_val = np.zeros(config.shape[0], dtype=c_dtype)[..., None]

        E0, _ = fori_loop(start, end, body_fun1, (start_val, config))
        E1, _, _ = fori_loop(start, end, body_fun2, (start_val, mask, config))
        logprobarr, _ = fori_loop(start, end, body_fun3, (logprobarr, config))
        vi_fliped, _ = fori_loop(start, end, body_fun4, (vi_fliped, config))
        # Can't use if statements in jitted code, need to use lax primitive instead.
        E0 = jax.lax.cond(pbc, E0, pbc_contrib1, E0, lambda E: np.add(E0, 0))
        E1 = jax.lax.cond(pbc, E1, pbc_contrib2, E1, lambda E: np.add(E1, 0))

        logprobarr = jax.ops.index_update(
            logprobarr, jax.ops.index[:, -1], amp_fliped(config, -1, 0)
        )

        vi_fliped = jax.ops.index_update(
            vi_fliped, jax.ops.index[:, -1], _vi_fliped(config, -1, 0)
        )

        E = E0 + E1

        return E, E0, E1, logpsi, logprobarr, mask, vi, vi_fliped

    return energy


def energy_sutherland_1d_init(log_amplitude, J, pbc, c_dtype):
    @jit
    def energy(net_params, config):
        @jit
        def amplitude_diff(config, i, j, idx):
            """compute apmplitude ratio of logpsi and logpsi_flipped, where i and i+1
            have their sign flipped. As logpsi returns the real and the imaginary part
            seperately, we therefor need to recombine them into a complex valued array"""
            config = jax.ops.index_update(config, jax.ops.index[:, i], idx[:, 0])
            config = jax.ops.index_update(config, jax.ops.index[:, j], idx[:, 1])
            logpsi_swaped = log_amplitude(net_params, config)
            logpsi_swaped = logpsi_swaped[0] + logpsi_swaped[1] * 1j
            return np.exp(logpsi_swaped - logpsi)

        @jit
        def body_fun(i, loop_carry):
            E, s = loop_carry
            idx = s[:, [i + 1, i]]
            E += amplitude_diff(config, i, i + 1, idx)
            return E, s

        def pbc_contrib(E):
            idx = config[:, [0, -1]]
            E += amplitude_diff(config, -1, 0, idx)
            return E

        logpsi = log_amplitude(net_params, config)
        logpsi = logpsi[0] + logpsi[1] * 1j
        logpsi = logpsi.astype(c_dtype)

        start = 0
        end = config.shape[1] - 1
        start_val = np.zeros(config.shape[0], dtype=c_dtype)[..., None]

        E, _ = fori_loop(start, end, body_fun, (start_val, config))

        # Can't use if statements in jitted code, need to use lax primitive instead.
        E = jax.lax.cond(pbc, E, pbc_contrib, E, lambda E: np.add(E, 0))

        E0 = E
        E1 = E
        return E, E0, E1, logpsi

    return energy


@jit
def energy_var(energy):
    return np.var(energy.real)


@jit
def magnetization(config):
    mag = np.sum(config, axis=1)
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
