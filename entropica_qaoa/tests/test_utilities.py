"""
Test some of the functions together
"""
import numpy as np
import networkx as nx

from pyquil.paulis import PauliSum, sZ
from pyquil.quil import QubitPlaceholder

from entropica_qaoa.utilities import (random_hamiltonian,
                                      distances_dataset,
                                      gaussian_2Dclusters,
                                      hamiltonian_from_hyperparams,
                                      graph_from_hamiltonian,
                                      graph_from_hyperparams,
                                      hamiltonian_from_graph,
                                      random_k_regular_graph,
                                      plot_graph,
                                      hamiltonian_from_distances,
                                      ring_of_disagrees)

from entropica_qaoa.qaoa.parameters import StandardParams
from entropica_qaoa.qaoa.cost_function import QAOACostFunctionOnWFSim

q1, q2 = QubitPlaceholder(), QubitPlaceholder()
reg = [0, 1, 3, q1, q2]
singles = [0, 3, q1]
biases = [0.4, -0.3, 1.2]
pairs = [(0, 1), (3, q1), (q2, 0)]
couplings = [1.2, 3.4, 4.5]

single_terms = [c * sZ(q) for c, q in zip(biases, singles)]
pair_terms = [c * sZ(q[0]) * sZ(q[1]) for c, q in zip(couplings, pairs)]
hamiltonian = PauliSum([*single_terms, *pair_terms])

graph = nx.Graph()
for node, weight in zip(singles, biases):
    graph.add_node(node, weight=weight)
for edge, weight in zip(pairs, couplings):
    graph.add_edge(edge[0], edge[1], weight=weight)


def edgematch(e1, e2):
    return np.allclose(e1["weight"], e2["weight"])


def nodematch(n1, n2):
    try:
        return np.allclose(n1["weight"], n2["weight"])
    except KeyError:
        return n1 == n2


def test_hamiltonian_from_hyperparams():
    ham = hamiltonian_from_hyperparams(reg, singles, biases, pairs, couplings)
    assert ham == hamiltonian


def text_graph_from_hyperparams():
    G = graph_from_hyperparams(reg, singles, biases, pairs, couplings)
    assert nx.is_isomorphic(graph, G,
                            edge_match=edgematch, node_match=nodematch)


def test_graph_from_hamiltonian():
    G = graph_from_hamiltonian(hamiltonian)
    assert nx.is_isomorphic(graph, G,
                            edge_match=edgematch, node_match=nodematch)


def test_hamiltonian_from_graph():
    ham = hamiltonian_from_graph(graph)
    assert ham == hamiltonian


def test_random_hamiltonian():
    ham = random_hamiltonian(reg)
    print(ham)


def test_random_k_regular_graph_and_plot_graph():
    G = random_k_regular_graph(2, reg, seed=42, weighted=True, biases=True)
    plot_graph(G)


def test_hamiltonian_from_distances():
    dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    ham1 = hamiltonian_from_distances(
            dist, biases={i: i + 1 for i in range(3)})
    single_terms = 1 * sZ(0), 2 * sZ(1), 3 * sZ(2)
    coupling_terms = 1 * sZ(0) * sZ(1), 1*sZ(1) * sZ(2), 2 * sZ(0) * sZ(2)
    ham2 = PauliSum([*single_terms, *coupling_terms])
    assert ham1 == ham2


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

def test_ring_of_disagrees():
    
    """
    Tests the known analytic solution to p=1 QAOA of the Ring of Disagrees
    See https://arxiv.org/pdf/1805.03265.pdf (section 5.4.5)
    """
    
    n_qubits = 8
    hamiltonian = ring_of_disagrees(n_qubits)
    p = 1
    
    # Known optimal angles
    betas = [np.pi/8]
    gammas = [np.pi/4]
    
    params = (betas,gammas)
    params_std = StandardParams([hamiltonian, p], params) 
    
    cost_fn = QAOACostFunctionOnWFSim(hamiltonian, params=params_std)
    exp_val = cost_fn(params_std.raw())

    assert np.isclose(exp_val, -6)

