import jax
from jax import vmap, jit, random, config
import jax.numpy as np
from jax.lax import fori_loop

import flax

from .net import conv, lstm
from .sampler import sample_init
from .optim import step_init
from .wavefunction import log_amplitude

import matplotlib.pyplot as plt


def initialize_model_1d(
    hamiltonian,
    width,
    filter_size,
    seed,
    num_spins,
    lr,
    J,
    J2,
    batch_size,
    hidden_size,
    depth,
    pbc,
    network,
    x64=False,
    one_hot=True,
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

    net_dispatch = {"conv": conv, "lstm": lstm}

    energy_dispatch = {
        "ising1d": energy_ising_1d_init,
        "heisenberg1d": energy_heisenberg_1d_init,
        "J1J21d": energy_J1J2_1d_init,
    }

    key = random.PRNGKey(seed)
    key, subkey, carry_key = random.split(key, 3)

    try:
        net = net_dispatch[network]
        if network == "conv":
            module = net.partial(
                depth=depth, features=width, kernel_size=filter_size, use_one_hot=True
            )
        elif network == "lstm":
            module = net.partial(
                hidden_size=hidden_size, key=carry_key, depth=depth, use_one_hot=one_hot
            )

    except KeyError:
        print(
            f"{network} is not a valid network. You can choose between small_net_1d and small_resnet_1d."
        )
        raise

    try:
        energy_init = energy_dispatch[hamiltonian]
        if hamiltonian == "J1J21d":
            energy_dict = {"J1": J, "J2": J2, "pbc": pbc, "c_dtype": c_dtype}
        else:
            energy_dict = {"J": J, "pbc": pbc, "c_dtype": c_dtype}
    except KeyError:
        print(
            f"{hamiltonian} is not a valid hamiltonian. You can choose between ising1d and heisenberg1d."
        )
        raise

    if one_hot:
        in_shape = (batch_size, num_spins, 2)
    else:
        in_shape = (batch_size, num_spins, 1)

    _, params = module.init_by_shape(subkey, [in_shape])
    model = flax.nn.Model(module, params)

    init_config = np.zeros((batch_size, num_spins, 1), dtype=f_dtype)
    sample_fn = sample_init(init_config)
    energy_fn = energy_init(**energy_dict)
    optimizer = flax.optim.Adam(learning_rate=lr).create(model)
    step = step_init(energy_fn, sample_fn, energy_var, magnetization)
    return (step, key, energy_fn, sample_fn, init_config, optimizer)


def energy_ising_1d_init(J=None, pbc=None, c_dtype=None):
    @jit
    def energy(model, config):
        @jit
        def amplitude_diff(config, i):
            """Compute amplitude ratio of logpsi and logpsi_flipped, where spin i has its
            sign flipped."""
            flipped = jax.ops.index_mul(config, jax.ops.index[:, i], -1)
            logpsi_flipped = log_amplitude(model, flipped)
            return np.exp(logpsi_flipped - logpsi)

        @jit
        def body_fun(i, loop_carry):
            E, s = loop_carry
            E -= J * (amplitude_diff(s, i) + s[:, i] * s[:, (i + 1) % N])
            return E, s

        logpsi = log_amplitude(model, config)
        logpsi = logpsi.astype(c_dtype)

        start = 0
        B, N, _ = config.shape
        # Can't use if statements in jitted code, need to use lax primitive instead.
        end = jax.lax.cond(pbc, N, lambda x: x, N - 1, lambda x: x)
        start_val = np.zeros(B, dtype=c_dtype)[..., None]

        E, _ = fori_loop(start, end, body_fun, (start_val, config))
        E = jax.lax.cond(
            pbc, E, lambda x: x, E, lambda x: x - amplitude_diff(config, -1)
        )
        return E

    return energy


def energy_heisenberg_1d_init(J=None, pbc=None, c_dtype=None):
    @jit
    def energy(model, config):
        @jit
        def amplitude_diff(config, i):
            """compute apmplitude ratio of logpsi and logpsi_flipped, where i and i+1
            have their sign flipped."""
            flipped = jax.ops.index_mul(config, jax.ops.index[:, i], -1)
            flipped = jax.ops.index_mul(flipped, jax.ops.index[:, (i + 1) % N], -1)
            logpsi_flipped = log_amplitude(model, flipped)
            return np.exp(logpsi_flipped - logpsi)

        @jit
        def body_fun(i, loop_carry):
            E, m, s = loop_carry
            E += (
                J
                * 0.25
                * (m[:, i] * amplitude_diff(s, i) + s[:, i] * s[:, (i + 1) % N])
            )
            return E, m, s

        # sx*sx + sy*sy gives a contribution iff x[i]!=x[i+1]
        mask = config * np.roll(config, -1, axis=1) - 1
        logpsi = log_amplitude(model, config)

        start = 0
        B, N, _ = config.shape
        # Can't use if statements in jitted code, need to use lax primitive instead.
        end = jax.lax.cond(pbc, N, lambda x: x, N - 1, lambda x: x)

        start_val = np.zeros(B, dtype=c_dtype)[..., None]

        E, _, _ = fori_loop(start, end, body_fun, (start_val, mask, config))
        return E

    return energy


def energy_J1J2_1d_init(J1=None, J2=None, pbc=None, c_dtype=None):
    @jit
    def energy(model, config):
        @jit
        def amplitude_diff(config, i, k):
            """compute apmplitude ratio of logpsi and logpsi_flipped, where i and i+k
            have their sign flipped."""
            flipped = jax.ops.index_mul(config, jax.ops.index[:, i], -1)
            flipped = jax.ops.index_mul(flipped, jax.ops.index[:, (i + k) % N], -1)
            logpsi_flipped = log_amplitude(model, flipped)
            return np.exp(logpsi_flipped - logpsi)

        @jit
        def body_fun1(i, loop_carry):
            E, m, s = loop_carry
            E += (
                J1
                * 0.25
                * (m[:, i] * amplitude_diff(s, i, 1) + s[:, i] * s[:, (i + 1) % N])
            )
            return E, m, s

        @jit
        def body_fun2(i, loop_carry):
            E, m, s = loop_carry
            E += (
                J2
                * 0.25
                * (m[:, i] * amplitude_diff(s, i, 2) + s[:, i] * s[:, (i + 2) % N])
            )
            return E, m, s

        # sx*sx + sy*sy gives a contribution iff x[i]!=x[i+1]
        mask1 = config * np.roll(config, -1, axis=1) - 1
        mask2 = 1 - config * np.roll(config, -2, axis=1)
        logpsi = log_amplitude(model, config)

        start = 0
        B, N, _ = config.shape
        # Can't use if statements in jitted code, need to use lax primitive instead.
        end1 = jax.lax.cond(pbc, N, lambda x: x, N - 1, lambda x: x)
        end2 = jax.lax.cond(pbc, N, lambda x: x, N - 2, lambda x: x)

        start_val = np.zeros(B, dtype=c_dtype)[..., None]

        E1, _, _ = fori_loop(start, end1, body_fun1, (start_val, mask1, config))
        E2, _, _ = fori_loop(start, end2, body_fun2, (start_val, mask2, config))
        return E1 + E2

    return energy


@jit
def energy_var(energy):
    return np.var(energy.real)


@jit
def magnetization(config):
    mag = np.sum(config, axis=1)
    return np.mean(mag)


@jit
def SzSz(config, i, j):
    def mul(i, j):
        return np.mean(0.25 * config[:, i] * config[:, j])

    mul = vmap(mul, in_axes=(0, None))
    mul = vmap(mul, in_axes=(None, 0))
    return mul(np.array(i), np.array(j))


@jit
def SxSx(model, config, i, j):
    @jit
    def amplitude_diff(i, j):
        """compute apmplitude ratio of logpsi and logpsi_flipped, where i and j
        have their sign flipped."""
        flipped = jax.ops.index_mul(config, jax.ops.index[:, [i, j]], -1)
        logpsi_flipped = log_amplitude(model, flipped)
        return np.mean(np.real(0.25 * np.exp(logpsi_flipped - logpsi)))

    logpsi = log_amplitude(model, config)
    amplitude_diff = vmap(amplitude_diff, in_axes=(0, None))
    amplitude_diff = vmap(amplitude_diff, in_axes=(None, 0))
    return amplitude_diff(np.array(i), np.array(j))


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
