import setuptools

INSTALL_REQUIRES = ["absl-py", "numpy", "jax>=0.1.55", "jaxlib>=0.1.37"]

setuptools.setup(
    name="nqs",
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),
    description="Neural Quantum States implemented in JAX.",
    python_requires=">=3.7",
)
