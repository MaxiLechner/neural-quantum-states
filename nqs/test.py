print(__name__)

try:
    # Trying to find module on sys.path
    import hamiltonian

    print("absolute")
    # print("absolute: ", hamiltonian.debug)
except ModuleNotFoundError:
    print("Absolute import failed")

try:
    # Trying to find module in the parent package
    from . import hamiltonian

    print("relative")
    # print("relative: ", hamiltonian.debug)
    del hamiltonian
except ImportError:
    print("Relative import failed")
