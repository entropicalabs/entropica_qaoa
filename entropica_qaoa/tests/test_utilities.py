"""
Test some of the functions together
"""

import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator, local_qvm, get_qc

from entropica_qaoa.utilities import (random_hamiltonian,
                                      distances_dataset,
                                      gaussian_2Dclusters)


def test_random_hamiltonian():
    nqubits = 2
    ham = random_hamiltonian(nqubits)
    print(ham)


def test_distances_dataset():
    data = [[1.5, 2.0], [3, 4], [6, 5], [10, 1]]
    print(distances_dataset(data))


def test_Gaussian_clusters():
    n_clusters = 3
    n_points = [10, 10, 10]
    means = [[0, 0], [1, 1], [1, 3]]
    cov_matrices = [np.array([[1, 0], [0, 1]]),
                    np.array([[0.5, 0], [0, 0.5]]),
                    np.array([[0.5, 0], [0, 0.5]])
                    ]

    data = gaussian_2Dclusters(n_clusters, n_points, means,
                                      cov_matrices)
    print(data)
