# Communities in the Kuramoto model

This repository contains the code for the preprint:
Tâm J Nguyên, Darrick Lee, Bernadette J Stolz
Communities in the Kuramoto model: dynamics and detection via path signatures, preprint

contact:
[tam.nguyen@epfl.ch](mailto:tam.nguyen@epfl.ch)

## Installation

In order to setup the environment using conda, you can use the following command,

```bash
conda create --name ksbm-path-signatures --file requirements.txt
conda activate ksbm-path-signatures
```

The libraries required to run the model (not obtain the figures) are:
* numpy
* scipy (ODE solver)
* sympy (permutations for agreement)

## File Structure

* kuramoto.py: module to simulate a generalized kuramoto model
* matrix_generation.py: module to generate stochastic block model (SBM) adjacency matrices
* signature.py: module to compute the signatures up to level two and lead matrices (original code by Léo Levy)
* display.py: utility module to display matrices, network and states for the kuramoto model

* XXX KSBM.ipynb: time-series, critical time, lead and covariance matrices, mean-field and gaussian KSBM, community estimations associated with Standard, Collapsed, Noisy and Large KSBM configurations. (Figures 4-13 with the exception of Figure 5)
* Critical Times.ipynb: dominated & identical gaussian KSBM and critical times for various kappa (Figure 5)

The structural community estimation algorithm is implemented in the jupyter notebooks as the function,
```bash
identify(M)
```
where $M$ is any block clustered matrix of interest.

## Quick-Start

Running the KSBM (here with the Standard KSBM parameters)

```bash
import matrix_generation as mat
from kuramoto import Kuramoto

N = 99                          # total number of oscillators
n = 3                           # number of communities
kappa = 100                     # coupling strength
A = (kappa/N)*fc_SBM_mat        # coupling matrix
freq = 2*np.array([1, 2, 3])/n  # (mean) intrinsic frequencies [rad/s]


fc_SBM_mat = mat.fully_connected_SBM(N, n) # sampling an assortative SBM

t = np.linspace(0, 10, 500)
model = Kuramoto(N, A, mu=0, sigma=0.1, frequencies=np.repeat(freq, N//n))
sol = model.simulate(t).y # time-serie of oscillators' phase in time [N x 500]
```

Computing lead matrices

```bash
from signature import lead_matrix

path = sol[:, :]
sin_path = np.sin(path)

L = lead_matrix(path)
L_sin = lead_matrix(sin_path)
```

## Citation

Will be provided here as soon as we upload the preprint
