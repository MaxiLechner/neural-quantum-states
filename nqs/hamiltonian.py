import jax
from jax import random, config
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
    x64=False,
):
    if config.read("jax_enable_x64") and x64:
        f_dtype = np.float64
        c_dtype = np.complex128
    elif not config.read("jax_enable_x64") and not x64:
        f_dtype = np.float32
        c_dtype = np.complex64
    else:
        raise Exception(
            """To use x32/x64 mode, both the variable x64 and the environment variable
            jax_enable_x64 have to agree. Setting the latter variable is described in
            https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision."""
        )

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
        print(
            f"{hamiltonian} is not a valid hamiltonian. You can choose between ising1d and heisenberg1d."
        )
        raise

    model = net(width, filter_size, net_dtype=f_dtype)
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
    return step, opt_state, key


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
            "Contribution due to periodic boundary condition."
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

        return E

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
        def body_fun(i, loop_carry):
            E, m, s = loop_carry
            E += (
                J
                * 0.25
                * (m[:, i] * amplitude_diff(s, i, i + 1) + s[:, i] * s[:, i + 1])
            )
            return E, m, s

        @jit
        def pbc_contrib(E):
            "Contribution due to periodic boundary condition."
            E += (
                J
                * 0.25
                * (
                    mask[:, -1] * amplitude_diff(config, -1, 0)
                    + config[:, -1] * config[:, 0]
                )
            )
            return E

        # sx*sx + sy*sy gives a contribution iff x[i]!=x[i+1]
        mask = config * np.roll(config, -1, axis=1) - 1
        logpsi = log_amplitude(net_params, config)
        logpsi = logpsi[0] + logpsi[1] * 1j

        start = 0
        end = config.shape[1] - 1
        start_val = np.zeros(config.shape[0], dtype=c_dtype)[..., None]

        E, _, _ = fori_loop(start, end, body_fun, (start_val, mask, config))

        # Can't use if statements in jitted code, need to use lax primitive instead.
        E = jax.lax.cond(pbc, E, pbc_contrib, E, lambda x: x)

        return E

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
            "Contribution due to periodic boundary condition."
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
