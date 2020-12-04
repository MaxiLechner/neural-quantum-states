import jax
from jax import vmap, jit, random, config
import jax.numpy as jnp

import flax

from .networks import conv, lstm
from .optim import step_init
from .wavefunction import log_amplitude

from functools import partial


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
        dtype = jnp.float64
    elif not config.read("jax_enable_x64") and not x64:
        dtype = jnp.float32
    else:
        raise Exception(
            """To use x32/x64 mode, both the variable x64 and the environment variable
            jax_enable_x64 have to agree. Setting the latter variable is described in
            https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Double-(64bit)-precision."""
        )

    net_dispatch = {"conv": conv, "lstm": lstm}

    energy_dispatch = {
        "ising_1d": energy_ising_1d_init,
        "heisenberg_1d": energy_heisenberg_1d_init,
        "J1J2_1d": energy_J1J2_1d_init,
    }

    key = random.PRNGKey(seed)
    key, subkey = random.split(key, 2)
    init_config = jnp.zeros((batch_size, num_spins, 1), dtype=dtype)
    end = num_spins if pbc else num_spins - 1
    end2 = num_spins if pbc else num_spins - 2

    try:
        net = net_dispatch[network]
        if network == "conv":
            module = net.partial(
                depth=depth,
                features=width,
                kernel_size=filter_size,
                use_one_hot=one_hot,
                init_config=init_config,
            )
        elif network == "lstm" or network == "is_lstm":
            module = net.partial(
                hidden_size=hidden_size,
                depth=depth,
                use_one_hot=one_hot,
                init_config=init_config,
            )

    except KeyError:
        print(
            f"{network} is not a valid network. You can choose between conv and lstm."
        )
        raise

    try:
        energy_init = energy_dispatch[hamiltonian]
        if hamiltonian == "J1J2_1d":
            energy_dict = {
                "J1": J,
                "J2": J2,
                "end1": end,
                "end2": end2,
            }
        else:
            energy_dict = {"J": J, "end": end}
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

    energy_fn = energy_init(**energy_dict)
    optimizer = flax.optim.LAMB(learning_rate=lr).create(model)
    step = step_init(energy_fn, energy_var, magnetization)
    return (step, key, energy_fn, init_config, optimizer)


def energy_ising_1d_init(J=None, end=None):
    @jit
    def energy(model, config):
        @jit
        def amplitude_diff(config, i):
            """Compute amplitude ratio of logpsi and logpsi_flipped, where spin i has its
            sign flipped."""
            flipped = jax.ops.index_mul(config, jax.ops.index[:, i], -1)
            logpsi_flipped = log_amplitude(model, flipped)
            return jnp.exp(logpsi_flipped - logpsi)

        vmap_amplitude_diff = vmap(partial(amplitude_diff, config), out_axes=1)

        logpsi = log_amplitude(model, config)

        _, N, _ = config.shape

        idx = jnp.arange(N)
        E0 = jnp.sum(config[:, :end] * jnp.roll(config, -1, axis=1)[:, :end], axis=1)
        E1 = jnp.sum(vmap_amplitude_diff(idx), axis=1)

        E = -(E0 + J * E1)
        return E

    return energy


def energy_heisenberg_1d_init(J=None, end=None):
    @jit
    def energy(model, config):
        @jit
        def amplitude_diff(config, i):
            """compute amplitude ratio of logpsi and logpsi_flipped, where i and i+1
            have their sign flipped."""
            flipped = jax.ops.index_mul(config, jax.ops.index[:, [i, (i + 1) % N]], -1)
            logpsi_flipped = log_amplitude(model, flipped)
            return jnp.exp(logpsi_flipped - logpsi)

        vmap_amplitude_diff = vmap(partial(amplitude_diff, config), out_axes=1)

        logpsi = log_amplitude(model, config)

        _, N, _ = config.shape

        idx = jnp.arange(end)
        # sz*sz term
        nn = config[:, :end] * jnp.roll(config, -1, axis=1)[:, :end]
        # sx*sx + sy*sy gives a contribution iff x[i]!=x[i+1]
        mask = nn - 1

        E0 = jnp.sum(nn, axis=1)
        E1 = jnp.sum(mask * vmap_amplitude_diff(idx), axis=1)

        E = 0.25 * J * (E0 + E1)

        return E

    return energy


def energy_J1J2_1d_init(J1=None, J2=None, end1=None, end2=None):
    @jit
    def energy(model, config):
        @jit
        def amplitude_diff(i, k):
            """compute amplitude ratio of logpsi and logpsi_flipped, where i and i+k
            have their sign flipped."""
            flipped = jax.ops.index_mul(config, jax.ops.index[:, [i, (i + k) % N]], -1)
            logpsi_flipped = log_amplitude(model, flipped)
            return jnp.exp(logpsi_flipped - logpsi)

        amplitude_diff = vmap(amplitude_diff, in_axes=(0, None), out_axes=1)

        logpsi = log_amplitude(model, config)

        _, N, _ = config.shape

        idx1 = jnp.arange(end1)
        idx2 = jnp.arange(end2)

        # sz*sz term
        nn1 = config[:, :end1] * jnp.roll(config, -1, axis=1)[:, :end1]
        nn2 = config[:, :end2] * jnp.roll(config, -2, axis=1)[:, :end2]
        # sx*sx + sy*sy gives a contribution iff x[i]!=x[i+1]
        mask1 = nn1 - 1
        mask2 = 1 - nn2

        E0_J1 = jnp.sum(nn1, axis=1)
        E0_J2 = jnp.sum(nn2, axis=1)
        E1_J1 = jnp.sum(mask1 * amplitude_diff(idx1, 1), axis=1)
        E1_J2 = jnp.sum(mask2 * amplitude_diff(idx2, 2), axis=1)

        E_J1 = 0.25 * J1 * (E0_J1 + E1_J1)
        E_J2 = 0.25 * J2 * (E0_J2 + E1_J2)
        return E_J1 + E_J2

    return energy


@jit
def energy_var(energy):
    return jnp.var(energy.real)


@jit
def magnetization(config):
    mag = jnp.sum(config, axis=1)
    return jnp.mean(mag)


@jit
def SzSz(config, i, j):
    def mul(i, j):
        return jnp.mean(0.25 * config[:, i] * config[:, j])

    mul = vmap(mul, in_axes=(0, None))
    mul = vmap(mul, in_axes=(None, 0))
    return mul(jnp.array(i), jnp.array(j))


@jit
def SxSx(model, config, i, j):
    @jit
    def amplitude_diff(i, j):
        """compute amplitude ratio of logpsi and logpsi_flipped, where i and j
        have their sign flipped."""
        flipped = jax.ops.index_mul(config, jax.ops.index[:, [i, j]], -1)
        logpsi_flipped = log_amplitude(model, flipped)
        return jnp.mean(jnp.real(0.25 * jnp.exp(logpsi_flipped - logpsi)))

    logpsi = log_amplitude(model, config)
    amplitude_diff = vmap(amplitude_diff, in_axes=(0, None))
    amplitude_diff = vmap(amplitude_diff, in_axes=(None, 0))
    return amplitude_diff(jnp.array(i), jnp.array(j))
