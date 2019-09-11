from jax import random
import jax.numpy as np
from network import net
from wavefunction import psi

# import numpy as onp
# import pdb

if __name__ == "__main__":
    """define network and input parameters"""
    N = 20  # input size
    M = (
        2 * 2
    )  # number of possible values each input can take times 2 as lax.conv only works with real valued weights
    B = 5  # batchsize
    data = np.zeros((B, N, 1), dtype=np.float32)
    FilterSize = 5  # width of conv, (FilterSize-1)/2 nonzero elements, must be odd
    model = net(M, FilterSize)
    ########################################################################################
    net_init, net_apply = model
    key = random.PRNGKey(1)
    key, subkey = random.split(key)
    in_shape = (1, N, 1)
    out_shape, net_params = net_init(subkey, in_shape)
    # print(net_params)
    # print(out_shape)
    net_apply = net_apply
    vi = net_apply(net_params, data)
    ps = psi(vi)
    print(ps)
