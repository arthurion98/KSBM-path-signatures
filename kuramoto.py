import numpy as np
from scipy.integrate import solve_ivp


class Kuramoto:
    def __init__(self, N, A=None, mu=0, sigma=1, frequencies=None, seed=1998):
        """Generalized Kuramoto model. This class allows for simulation of a Kuramoto model with
        arbitrary adjacency matrix along a specified time discretization.

        parameters:
            N (int): Number of oscillators.
            A (np.ndarray [N x N], optional): Adjacency matrix. Defaults to the (non-normalized) standard Kuramoto model with kappa=1.
            mu (int, optional): Mean of the noise to intrinsic frequencies. Defaults to 0.
            sigma (int, optional): Standard deviation of the noise to intrinsic frequencies. Defaults to 1.
            frequencies (np.ndarray, optional): Frequencies of the oscillators (to which noise is added). Defaults to noise.
            seed (int, optional): Seed for reproducibility. 
        """

        self.seed = seed
        np.random.seed(self.seed)
        self.N = N
        self.A = np.ones((N, N)) - np.eye(N) if A is None else A
        noise = sigma * np.random.randn(self.N) + mu
        self.frequencies = noise if frequencies is None else frequencies + noise
        self.thetas_0 = np.random.rand(self.N) * 2 * np.pi

    def reinitialize(self, mu=0, sigma=1):  # used to restart an instance
        self.frequencies = sigma * np.random.randn(self.N) + mu
        self.thetas_0 = np.random.rand(self.N) * 2 * np.pi

    def time_diff(self, t, thetas):
        diff_thetas = np.array([w + np.sum(self.A[i]*np.sin(thetas - thetas[i])) for w, i in zip(self.frequencies, range(self.N))])
        return diff_thetas

    def simulate(self, t):
        sol = solve_ivp(self.time_diff, [t[0], t[-1]], self.thetas_0, t_eval=t)
        return sol
