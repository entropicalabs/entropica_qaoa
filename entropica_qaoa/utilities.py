#   Copyright 2019 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Utility and convenience functions for a number of QAOA applications.
"""

from typing import Union, List, Dict, Iterable, Tuple
import random
import itertools

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

from pyquil import Program
from pyquil.quil import QubitPlaceholder
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.gates import X
from scipy.spatial import distance

#############################################################################
# METHODS FOR CREATING HAMILTONIANS AND GRAPHS, AND SWITCHING BETWEEN THE TWO
#############################################################################


def hamiltonian_from_hyperparams(reg: Iterable[Union[int, QubitPlaceholder]],
                                 singles: List[int],
                                 biases: List[float],
                                 pairs: List[Tuple[int, QubitPlaceholder]],
                                 couplings: List[float]) -> PauliSum:
    """
    Builds a cost Hamiltonian as a PauliSum from a specified set of problem hyperparameters.

    Parameters
    ----------
    reg:
        The register to apply the beta rotations on.
    singles:
        The register indices of the qubits that have a bias term
    biases:
        Values of the biases on the qubits specified in singles.
    pairs:
        The qubit pairs that have a non-zero coupling coefficient.
    couplings:
        The value of the couplings for each pair of qubits in pairs.

    Returns
    -------
    Hamiltonian
        The PauliSum built from these hyperams.
    """
    hamiltonian = []
    for pair, coupling in zip(pairs, couplings):
        hamiltonian.append(PauliTerm('Z', pair[0], coupling) *
                           PauliTerm('Z', pair[1]))

    for single, bias in zip(singles, biases):
        hamiltonian.append(PauliTerm('Z', single, bias))

    return PauliSum(hamiltonian)


def graph_from_hyperparams(reg: List[Union[int, QubitPlaceholder]],
                           singles: List[int],
                           biases: List[float],
                           pairs: List[int],
                           couplings: List[float]) -> nx.Graph:

    """
    Builds a networkx graph from the specified QAOA hyperparameters

    Parameters
    ----------
    nqubits:
        The number of qubits (graph nodes)
    singles:
        The qubits that have a bias term (node weight)
    biases:
        The values of the single-qubit biases (i.e. the node weight values)
    pairs:
        The qubit pairs that are coupled (i.e. the nodes conected by an edge)
    couplings:
        The strength of the coupling between the qubit pairs (i.e. the edge weights)

    Returns
    -------
    nx.Graph:
        Networkx graph with the specified properties

    """

    G = nx.Graph()

    for qubit, weight in zip(singles, biases):
        G.add_node(qubit, weight=weight)

    for pair, weight in zip(pairs, couplings):
        G.add_edge(pair[0], pair[1], weight=weight)

    return G


def random_hamiltonian(reg: List[Union[int, QubitPlaceholder]]) -> PauliSum:
    """
    Creates a random cost hamiltonian, diagonal in the computational basis:

     - Randomly selects which qubits that will have a bias term, then assigns
       them a bias coefficient.
     - Randomly selects which qubit pairs will have a coupling term, then
       assigns them a coupling coefficient.

     In both cases, the random coefficient is drawn from the uniform
     distribution on the interval [0,1).

    Parameters
    ----------
    reg:
        register to build the hamiltonian on.

    Returns
    -------
    PauliSum:
        A hamiltonian with random couplings and biases, as a PauliSum object.

    """
    hamiltonian = []

    n_biases = np.random.randint(len(reg))
    bias_qubits = random.sample(reg, n_biases)
    bias_coeffs = np.random.rand(n_biases)

    for qubit, coeff in zip(bias_qubits, bias_coeffs):
        hamiltonian.append(PauliTerm("Z", qubit, coeff))

    for q1, q2 in itertools.combinations(reg, 2):
        are_coupled = np.random.randint(2)
        if are_coupled:
            couple_coeff = np.random.rand()
            hamiltonian.append(PauliTerm("Z", q1, couple_coeff) *
                               PauliTerm("Z", q2))

    return PauliSum(hamiltonian)


def graph_from_hamiltonian(hamiltonian: PauliSum) -> nx.Graph:
    """
    Creates a networkx graph corresponding to a specified problem Hamiltonian.

    Parameters
    ----------
    hamiltonian:
        The Hamiltonian of interest. Must be specified as a PauliSum object.

    Returns
    -------
    G:
        The corresponding networkx graph with the edge weights being the
        two-qubit coupling coefficients,
        and the node weights being the single-qubit bias terms.

    TODO:

        Allow ndarrays to be input as hamiltonian too.
        Provide support for qubit placeholders.

    """

    # Get hyperparameters from Hamiltonian

    reg = hamiltonian.get_qubits()
    hyperparams = {'reg': reg, 'singles': [],
                   'biases': [], 'pairs': [], 'couplings': []}

    for term in hamiltonian:

        qubits = term.get_qubits()

        if len(qubits) == 0:
            # Term is proportional to identity - doesn't act on any qubits
            continue
        if len(qubits) == 1:
            hyperparams['singles'] += qubits
            hyperparams['biases'] += [term.coefficient.real]

        elif len(qubits) == 2:
            hyperparams['pairs'].append(qubits)
            hyperparams['couplings'] += [term.coefficient.real]

        else:
            raise ValueError("For now we only support hamiltonians with "
                             "up to 2 qubit terms")

    G = graph_from_hyperparams(*hyperparams.values())
    return G


def hamiltonian_from_graph(G: nx.Graph) -> PauliSum:
    """
    Builds a cost Hamiltonian as a PauliSum from a specified networkx graph,
    extracting any node biases and edge weights.

    Parameters
    ----------
    G:
        The networkx graph of interest.

    Returns
    -------
    PauliSum:
        The PauliSum representation of the networkx graph.

    """

    hamiltonian = []

    # Node bias terms
    bias_nodes = [*nx.get_node_attributes(G, 'weight')]
    biases = [*nx.get_node_attributes(G, 'weight').values()]

    for node, bias in zip(bias_nodes, biases):
        hamiltonian.append(PauliTerm("Z", node, bias))

    # Edge terms
    edges = list(G.edges)
    edge_weights = [*nx.get_edge_attributes(G, 'weight').values()]

    for edge, weight in zip(edges, edge_weights):
        hamiltonian.append(PauliTerm("Z", edge[0], weight) *
                           PauliTerm("Z", edge[1]))

    return PauliSum(hamiltonian)


def random_k_regular_graph(degree: int,
                           nodes: List[Union[int, QubitPlaceholder]],
                           seed: int = None,
                           weighted: bool = False,
                           biases: bool = False) -> nx.Graph:
    """
    Produces a random graph with specified number of nodes, each having degree k.

    Parameters
    ----------
    degree:
        Desired degree for the nodes
    nodes:
        The node set of the graph. Can be anything that works as a qubit for
        PauliSums.
    seed:
        A seed for the random number generator
    weighted:
        Whether the edge weights should be uniform or different. If false, all weights are set to 1.
        If true, the weight is set to a random number drawn from the uniform distribution in the interval 0 to 1.
        If true, the weight is set to a random number drawn from the uniform
        distribution in the interval 0 to 1.
    biases:
        Whether or not the graph nodes should be assigned a weight.
        If true, the weight is set to a random number drawn from the uniform
        distribution in the interval 0 to 1.

    Returns
    -------
    nx.Graph:
        A graph with the properties as specified.

    """
    np.random.seed(seed=seed)

    # create a random regular graph on the nodes
    G = nx.random_regular_graph(degree, len(nodes), seed)
    nx.relabel_nodes(G, {i: n for i, n in enumerate(nodes)})

    for edge in G.edges():
        if not weighted:
            G[edge[0]][edge[1]]['weight'] = 1
        else:
            G[edge[0]][edge[1]]['weight'] = np.random.rand()

    if biases:
        for node in G.nodes():
            G.node[node]['weight'] = np.random.rand()

    return G


def plot_graph(G, ax=None):
    """
    Plots a networkx graph.

    Parameters
    ----------
    G:
        The networkx graph of interest.
    ax: Matplotlib axes object
        Defaults to None. Matplotlib axes to plot on.
    """

    weights = np.real([*nx.get_edge_attributes(G, 'weight').values()])
    pos = nx.shell_layout(G)

    nx.draw(G, pos, node_color='#A0CBE2', with_labels=True, edge_color=weights,
            width=4, edge_cmap=plt.cm.Blues, ax=ax)
    plt.show()


#############################################################################
# HAMILTONIANS AND DATA
#############################################################################


def hamiltonian_from_distances(dist, biases=None) -> PauliSum:
    """
    Generates a Hamiltonian from a distance matrix and a numpy array of single
    qubit bias terms where the i'th indexed value of in biases is applied to
    the i'th qubit.

    Parameters
    ----------
    dist:
        A 2-dimensional square matrix or Pandas DataFrame, where entries in row i, column j
        represent the distance between node i and node j. Assumed to be
        symmetric
    biases:
        A dictionary of floats, with keys indicating the qubits with bias
        terms, and corresponding values being the bias coefficients.

    Returns
    -------
    PauliSum:
        A PauliSum object modelling the Hamiltonian of the system
    """
    pauli_list = []
    m, n = dist.shape

    # allows tolerance for both matrices and dataframes
    if isinstance(dist, pd.DataFrame):
        dist = dist.values

    if biases:
        if not isinstance(biases, type(dict())):
            raise ValueError('biases must be of type dict()')
        for key in biases:
            term = PauliTerm('Z', key, biases[key])
            pauli_list.append(term)

    # pairwise interactions
    for i in range(m):
        for j in range(n):
            if i < j:
                term = PauliTerm('Z', i, dist[i][j]) * PauliTerm('Z', j)
                pauli_list.append(term)

    return PauliSum(pauli_list)


def distances_dataset(data: Union[np.array, pd.DataFrame, Dict],
                      metric='euclidean') -> Union[np.array, pd.DataFrame]:
    """
    Computes the distance between data points in a specified dataset,
    according to the specified metric (default is Euclidean).

    Parameters
    ----------
    data:
        The user's dataset, either as an array, dictionary, or a Pandas
        DataFrame
    metric:
        Type of metric to calculate the distances used in
        ``scipy.spatial.distance``

    Returns
    -------
    Union[np.array, pd.Dataframe]:
        If input is a dictionary or numpy array, output is a numpy array of
        dimension NxN, where N is the number of data points.
        If input is a Pandas DataFrame, the distances are returned in this format.
    """

    if isinstance(data, dict):
        data = np.concatenate(list(data.values()))
    elif isinstance(data, pd.DataFrame):
        return pd.DataFrame(distance.cdist(data, data, metric),
                            index=data.index, columns=data.index)
    return distance.cdist(data, data, metric)


def gaussian_2Dclusters(n_clusters: int,
                        n_points: int,
                        means: List[float],
                        cov_matrices: List[float]):
    """
    Creates a set of clustered data points, where the distribution within each
    cluster is Gaussian.

    Parameters
    ----------
    n_clusters:
        The number of clusters
    n_points:
        A list of the number of points in each cluster
    means:
        A list of the means [x,y] coordinates of each cluster in the plane
        i.e. their centre)
    cov_matrices:
        A list of the covariance matrices of the clusters

    Returns
    -------
    data
        A dict whose keys are the cluster labels, and values are a matrix of
        the with the x and y coordinates as its rows.

    TODO
        Output data as Pandas DataFrame?

    """
    args_in = [len(means), len(cov_matrices), len(n_points)]
    assert all(item == n_clusters for item in args_in),\
            "Insufficient data provided for specified number of clusters"

    data = {}
    for i in range(n_clusters):

        cluster_mean = means[i]

        x, y = np.random.multivariate_normal(cluster_mean, cov_matrices[i], n_points[i]).T
        coords = np.array([x, y])
        tmp_dict = {str(i): coords.T}
        data.update(tmp_dict)

    return data


def plot_cluster_data(data):
    """
    Creates a scatterplot of the input data specified
    """

    data_matr = np.concatenate(list(data.values()))
    plt.scatter(data_matr[:, 0], data_matr[:, 1])
    plt.show()


#############################################################################
# ANALYTIC & KNOWN FORMULAE
#############################################################################


def ring_of_disagrees(n: int) -> PauliSum:
    """
    Builds the cost Hamiltonian for the "Ring of Disagrees" described in the
    original QAOA paper (https://arxiv.org/abs/1411.4028),
    for the specified number of vertices n.

    Parameters
    ----------
    n:
        Number of vertices in the ring

    Returns
    -------
    PauliSum:
        The cost Hamiltonian representing the ring, as a PauliSum object.

    """

    hamiltonian = []
    for i in range(n - 1):
        hamiltonian.append(PauliTerm("Z", i, 0.5) * PauliTerm("Z", i + 1))
        hamiltonian.append(PauliTerm("I",i,-0.5))
    hamiltonian.append(PauliTerm("Z", n - 1, 0.5) * PauliTerm("Z", 0))
    hamiltonian.append(PauliTerm("I",n-1,-0.5))

    return PauliSum(hamiltonian)


##########################################################################
# OTHER MISCELLANEOUS
##########################################################################


def prepare_classical_state(reg: List[Union[int, QubitPlaceholder]],
                            state: List[bool]) -> Program:
    """
    Prepare a custom classical state for all qubits in the specified register reg.

    Parameters
    ----------
    reg:
        Register to apply the state preparation circuit on. E.g. a list of
        qubits
    state:
        A list of 0s and 1s which represent the starting state of the register, bit-wise.

    Returns
    -------
    Program
        Quil Program with a circuit in an initial classical state.
    """

    if len(reg) != len(state):
        raise ValueError("qubit state must be the same length as reg")

    p = Program()
    for qubit, s in zip(reg, state):
        # if int(s) == 0 we don't need to add any gates, since the qubit is in
        # state 0 by default
        if int(s) == 1:
            p.inst(X(qubit))
    return p


def max_probability_bitstring(probs):
    """
    Returns the lowest energy state of a QAOA run from the list of
    probabilities returned by pyQuil's Wavefunction.probabilities()method.

    Parameters
    ----------
    probs:
        A numpy array of length 2^n, returned by Wavefunction.probabilities()

    Returns
    -------
    int:
        A little endian list of binary integers indicating the lowest energy
        state of the wavefunction.
    """

    index_max = max(range(len(probs)), key=probs.__getitem__)
    string = '{0:0' + str(int(np.log2(len(probs)))) + 'b}'
    string = string.format(index_max)
    return [int(item) for item in string]


def cluster_accuracy(state, true_labels):
    """
    Prints informative statements comparing QAOA's returned bit string for MaxCut to the
    true (known) cluster labels.

    Parameters
    ----------
    state:
        A little-endian list of binary integers representing the lowest energy
        state of the wavefunction
    true_labels:
        A little-endian list of binary integers representing the true solution
        to the MAXCUT clustering problem.
    """
    print('True Labels of samples:', true_labels)
    print('Lowest QAOA State:', state)
    acc = [a == b for a, b in zip(state, true_labels)].count(True) / len(state)
    print('Accuracy of Original State:', acc * 100, '%')
    acc_c = 1 - acc
    print('Accuracy of Complement State:', acc_c * 100, '%')


def plot_probabilities(probabilities: Union[np.array, list],
                       energies: Union[np.array, list],
                       ax=None):
    """Makes a nice plot of the probabilities for each state and its energy

    Parameters
    ----------
    probabilities:
        The probabilites to find the state. Can be calculated via wavefunction.probabilities()
    energies:
        The energy of that state
    ax: matplotlib axes object
        The canvas to draw on
    """
    if ax is None:
        fig, ax = plt.subplots()
    # normalizing energies
    energies = np.array(energies)
    energies /= max(abs(energies))

    nqubits = int(np.log2(len(energies)))
    format_strings = '{0:0' + str(nqubits) + 'b}'

    # create labels
    labels = [r'$\left|' +
              format_strings.format(i) + r'\right>$' for i in range(len(probabilities))]
    y_pos = np.arange(len(probabilities))
    width = 0.35
    ax.bar(y_pos, probabilities, width, label=r'Probability')

    ax.bar(y_pos + width, -energies, width, label="-1 x Energy")
    ax.set_xticks(y_pos + width / 2, minor=False)
    ax.set_xticklabels(labels, minor=False)
    ax.set_xlabel("State")
    ax.grid(linestyle='--')
    ax.legend()
