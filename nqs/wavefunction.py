import jax.numpy as jnp
from jax import jit

from nqs.networks import one_hot


@jit
def log_amplitude(model, config):
    """compute logpsi for a batch of samples. As the network returns
    the amplitude for both up and down states we need to pick the
    right amplitude by indexing according to the input configuration"""

    out = model(config)
    exp = jnp.exp(out)
    norm = jnp.linalg.norm(exp, 2, axis=2, keepdims=True) ** 2

    # pick out the amplitude corresponding to the input configuration
    vi = out * one_hot(config)
    vi = jnp.sum(vi, axis=-1)
    vi = vi[..., jnp.newaxis]

    logpsi = vi - 0.5 * jnp.log(norm)
    logpsi = jnp.sum(logpsi, axis=1)
    return logpsi
