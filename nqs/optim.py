import jax.numpy as jnp
from jax import jit, grad, tree_multimap

from .wavefunction import log_amplitude


@jit
def loss(model, config, energy):
    energy -= jnp.mean(energy)
    energy = energy.conj()
    lpsi = log_amplitude(model, config)
    return 2 * jnp.mean(jnp.real(energy * lpsi))


def step_init(energy_fn, energy_var, magnetization):
    @jit
    def step(optimizer, key, lr, init_config):
        model = optimizer.target
        key, config = model.sample(key=key, init_config=init_config)
        energy = energy_fn(model, config)
        grad_loss = grad(loss)(model, config, energy)
        var = energy_var(energy)
        mag = magnetization(config)
        grad_loss = tree_multimap(lambda g: jnp.clip(g, -1, 1), grad_loss)
        opt_update = optimizer.apply_gradient(grad_loss, learning_rate=lr)
        return opt_update, key, energy, mag, var

    return step
