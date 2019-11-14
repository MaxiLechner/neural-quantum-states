from ising1d import initialize_ising1d, ising1d
from network import net

# from wavefunction import lpsi
from optim import train


if __name__ == "__main__":
    numSpins = 10
    batchSize = 100  # batchsize
    numIt = 1000
    lr = 1e-02
    net_apply, net_params, key, lpsi = initialize_ising1d(numSpins, batchSize, net)
    sample, energy, grad = ising1d(net_apply, net_params, key, N, B, lpsi)

    train(optimizer, lr, numIt, key, net_params, plotting=True)
