import jax
import jax.numpy as np
from jax import jit, random
from jax.lax import fori_loop


@jit
def prob(x):
    x = np.exp(x)
    norm = np.linalg.norm(x, 2, axis=2, keepdims=True)
    x = x / norm
    probs = np.square(np.abs(x))
    return probs


def sample_init(net_apply):
    @jit
    def sample(net_params, config, key):
        def body(i, loop_carry):
            key, config = loop_carry
            out = net_apply(net_params, config)
            probs = prob(out)
            key, subkey = random.split(key)
            sample = random.bernoulli(subkey, probs[:, i, 1]) * 2 - 1.0
            sample = sample[..., np.newaxis]
            config = jax.ops.index_update(config, jax.ops.index[:, i], sample)
            return key, config

        key, config = fori_loop(0, config.shape[1], body, (key, config))
        return key, config

    return sample
