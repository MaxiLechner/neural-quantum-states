import jax.numpy as np
from jax.lib import pytree


def real_to_complex(arr):
    _, _, s3 = arr.shape
    assert s3 == 4
    carr = arr[:, :, [0, 2]] * np.exp(arr[:, :, [1, 3]] * 1j)
    return carr


def compute_probs(arr):
    arr = real_to_complex(arr)
    tc = np.exp(arr)
    nc = np.linalg.norm(tc, 2, axis=2, keepdims=True)
    tc = tc / nc
    probs = np.square(np.abs(tc))
    return probs


def expand_dimensions(w, n):
    """expand dims of the first array by adding np.newaxis on the
    right side to match the dimension of the second array"""
    n = n.shape
    length = len(n) - 1
    shape = (n[0],) + (1,) * length
    return w.reshape(shape)


def apply_elementwise(eloc, jac):
    """applies the local energy in an elementwise fashion to the jacobian params,
    function modeled after tree_util functions like tree_map"""
    leaves, treedef = pytree.flatten(jac)
    out = []
    for i in leaves:
        i = expand_dimensions(eloc, i) * i
        i = i.real
        i = i.mean(axis=0)
        i = np.squeeze(i, axis=0)
        i = 2 * i
        out.append(i)
    return treedef.unflatten(out)


def lpsi(net_apply, net_params, data):
    """compute logpsi for a batch of samples. As the network returns
    the amplitude for both up and down states we need to pick the
    right amplitude by indexing according to the samples"""
    arr = net_apply(net_params, data)
    arr = real_to_complex(arr)
    tc = np.exp(arr)
    nc = np.linalg.norm(tc, 2, axis=2, keepdims=True) ** 2

    idx = (data + 1) / 2
    idx = idx.astype("int32")
    B, N, _ = data.shape
    splits = np.split(arr, B)
    splits = [i.reshape(N, 2) for i in splits]
    isplits = np.split(idx, B)
    isplits = [i.reshape(N) for i in isplits]
    vi = np.stack([splits[j][np.arange(N), isplits[j]] for j in range(B)]).reshape(
        B, N, 1
    )

    logpsi = vi - 0.5 * np.log(nc)
    logpsi = np.sum(logpsi, axis=1)
    return np.real(logpsi), np.imag(logpsi)


def make_complex(state):
    """turns the real valued state into complex form, function modeled after tree_util functions like tree_map"""
    a, b = state
    assert len(a) == len(b)
    leaves, treedef = pytree.flatten(a)
    leaves2, _ = pytree.flatten(b)
    out = []
    for i in range(len(leaves)):
        out.append(leaves[i] + leaves2[i] * 1j)
    return treedef.unflatten(out)
