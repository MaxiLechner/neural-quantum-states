import jax
from jax import random
from wavefunction import compute_probs


def sample_one_batch(net, key, data):
    for i in range(data.shape[1]):
        vi = net(data)
        probs = compute_probs(vi)
        key, subkey = random.split(key)
        sample = random.bernoulli(subkey, probs[:, i, 0]) * 2 - 1.0
        sample = sample.reshape(data.shape[0], 1)
        data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
    return data
