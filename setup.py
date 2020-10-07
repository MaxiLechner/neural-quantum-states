import setuptools

INSTALL_REQUIRES = ["absl-py", "numpy", "jax>=0.2.1", "jaxlib>=0.1.55", "flax>=0.2.2"]

setuptools.setup(
    name="nqs",
    version="0.0.2",
    license="Apache 2.0",
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),
    description="Neural Quantum States implemented in JAX.",
    python_requires=">=3.6.9",
)
