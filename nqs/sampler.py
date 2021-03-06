import jax
import jax.numpy as jnp
from jax import jit, random
from jax.lax import fori_loop


@jit
def prob(x):
    x = jnp.exp(x)
    norm = jnp.linalg.norm(x, 2, axis=2, keepdims=True)
    x = x / norm
    probs = jnp.square(jnp.abs(x))
    return probs


def sample_init(init_config):
    @jit
    def sample(model, key):
        def body(i, loop_carry):
            key, config = loop_carry
            out = model(config)
            probs = prob(out)
            key, subkey = random.split(key)
            sample = random.bernoulli(subkey, probs[:, i, 1]) * 2 - 1.0
            sample = sample[..., jnp.newaxis]
            config = jax.ops.index_update(config, jax.ops.index[:, i], sample)
            return key, config

        key, config = fori_loop(0, init_config.shape[1], body, (key, init_config))
        return key, config

    return sample
