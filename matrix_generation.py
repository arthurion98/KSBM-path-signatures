import numpy as np
import scipy.optimize


def SBM(N, n, p, seed=1998):
    """Stochastic Block Model (SBM). Generates a SBM according to a specified probability matrix.

    parameters:
        N (int): Number of oscillators.
        n (int): Number of communities (must divide N).
        p (np.ndarray [n x n]): Symmetric probability matrix.
        seed (int, optional): Seed for reproducibility.
    """

    np.random.seed(seed)

    if N % n > 0:
        raise AssertionError

    N_c = N//n
    A = np.zeros((N,N))

    for i in range(N):
        for j in range(i):
            r = i//N_c
            s = j//N_c

            if np.random.rand() <= p[r, s]:
                A[i, j] = 1

    return A + A.T


def fully_connected_SBM(N, n, seed=1998):
    """Variant of the SBM for the assortative KSBM. Generates a SBM with fully-connected communities and a
    single edge projected from each node to a node in a different community.

    parameters:
        N (int): Number of oscillators.
        n (int): Number of communities (must divide N).
        seed (int, optional): Seed for reproducibility.
    """

    np.random.seed(seed)

    N_c = N//n
    args = (np.ones((N_c, N_c)) - np.eye(N_c) for r in range(n))
    D = scipy.linalg.block_diag(*args)

    B = np.zeros((N, N))

    old_comm = -1  # init value force a first computation
    for i in range(N):
        comm = i // n

        if comm != old_comm:  # avoid re-computation while in the same community
            choices = np.arange(N)
            choices = choices[choices // n != comm]

        B[i, np.random.choice(choices)] = 1

        old_comm = comm

    return np.clip(D + B + B.T, 0, 1)
