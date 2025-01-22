import numpy as np

"""Path-Signatures. Computes the path-signatures up to level two, as well as lead matrices, of a path
by interpolation of time-points.

Original code in MATLAB by: Léo Levy, Kuramoto model and path-signatures, BSc Thesis EPFL, 2023
Adapted to Python by: Tâm Johan Nguyên
"""


def signature(path, level):
    """Path-Signatures. Computes the path-signatures at a specified level of a real time-discretized path.
    Remark that the signatures are independent of the time-interval between points of the path, hence
    only the discrete points of the paths in R^N need to be specified.

    parameters:
        path (np.ndarray [N x T]): (Real) coordinates in R^N of the path at each discretized time points.
        level (int): Level of the signature to compute (level <= 2).
    """

    if level == 0:  # constant
        return 1
    if level == 1:  # vector
        return path[:, -1]-path[:, 0]
    if level == 2:  # matrix
        N = path.shape[0]
        S = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    S[i, j] = np.sum((path[i]-path[i, 0])[1:]*(path[j, 1:]-path[j, :-1]))

        return S
    else:
        raise NotImplementedError


def lead_matrix(path, S=None):
    """Lead Matrix. Computes the lead matrix from a path or the level two signature associated with the path.

    parameters:
        path (np.ndarray [N x T]): (Real) coordinates in R^N of the path at each discretized time points.
        S (np.ndarray [N x N], optional): Level two signature of the path.
    """

    N = path.shape[0]
    A = np.zeros((N, N))

    if S is None:  # use explicit interpolation construction
        for i in range(N - 1):
            for j in range(i + 1, N):
                A[i, j] = path[j, 0]*path[i, -1] - path[i, 0]*path[j, -1] + np.sum(path[i, :-1]*path[j, 1:] - path[j, :-1]*path[i, 1:])

        A = 0.5 * (A - A.T)
    else:  # use A[i,j] = 0.5 * (S[i,j] - S[j,i])
        A = 0.5 * (S - S.T)

    return A
