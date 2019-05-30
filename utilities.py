import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from pyquil.paulis import PauliSum, PauliTerm 
from qaoa.parameters import QAOAParameterIterator



## METHODS FOR CREATING RANDOM HAMILTONIANS AND GRAPHS, AND SWITCHING BETWEEN THE TWO ##

def create_random_hamiltonian(nqubits, single_terms=True, pair_terms=True):
    """
    Description
    -----------
    Creates a random hamiltonian.

    Parameters
    ----------
    :param      nqubits:            The number of qubits
    :param      single_terms=True:  Create single qubit terms?
    :param      pair_terms=True:    Create two qubit terms?

    Returns
    -------
    :returns:   (PauliSum) a random diagonal hamiltonian on nqubits qubits
    
    TODO: Since only Hamiltonians with couplings are interesting, the pair_terms argument is a little 
    unnecessary. Should we make this function a little more interesting by, eg:
        
        - allowing the user to specify the sparsity of the couplings they may want, rather than
        us just randomly choosing for them?
        - allowing a mean and standard deviation of the number of couplings to be passed in, then generate
        an instance obeying those statistics? Could be useful for studying averages over many random graphs with certain
        distributions.
        - adding in the ability to specify some maximum node degree (or again some statistical measure)?
    
    
    """
    hamiltonian = []
    if single_terms:
        numb_biases = np.random.randint(nqubits)
        bias_qubits = np.random.choice(nqubits,numb_biases,replace=False)
        bias_coeffs = np.random.rand(numb_biases)
        for i in range(numb_biases):
            hamiltonian.append(PauliTerm("Z", bias_qubits[i], bias_coeffs[i]))

    if pair_terms:
        for i in range(nqubits):
            for j in range(i+1,nqubits):
                are_coupled = np.random.randint(2)
                if are_coupled:
                    couple_coeff = np.random.rand()    
                    hamiltonian.append(PauliTerm("Z", i, couple_coeff)*PauliTerm("Z", j, 1.0))

    return PauliSum(hamiltonian)

def create_networkx_graph(vertices,edge_weights):
    
    """
    Creates a networkx graph on specified number of vertices, with the specified edge_weights
    """
    
    G = nx.Graph()
    G.add_nodes_from(range(vertices))
    i_pointer = 0
    for i in range(vertices):  

        for j in range(i,vertices):
        
            weight = edge_weights[i_pointer] + j
            G.add_edge(i,j,weight=weight)
        
        i_pointer += vertices - i
        
    return G

def networkx_from_hamiltonian(vertex_pairs,edge_weights):
    
    """
    Creates a networkx graph on specified number of vertices, with the specified edge_weights
    """
    
    G = nx.Graph()
    
    for i in range(len(vertex_pairs)):
        G.add_edge(vertex_pairs[i][0],vertex_pairs[i][1],weight=edge_weights[i])
        
    return G

def plot_networkx_graph(G):
    
    """
    Takes in a networkx graph and plots it
    
    TODO: can we also take in the list of nodes with a bias, and somehow show this on the plot too?
    """
    
    weights = [*nx.get_edge_attributes(G,'weight').values()]
    pos = nx.shell_layout(G)
    
    nx.draw(G,pos,node_color='#A0CBE2',with_labels=True,edge_color=weights,
                 width=4, edge_cmap=plt.cm.Blues)
    plt.show()

def hamiltonian_from_networkx(G):
    
    """
    Builds a Hamiltonian from a networkx graph
    """
    
    hamiltonian = []
    edges = list(G.edges)
    weights = [*nx.get_edge_attributes(G,'weight').values()]
    hamiltonian = []
    for i in range(len(edges)):
        hamiltonian.append(PauliTerm("Z", edges[i][0], weights[i])*PauliTerm("Z", edges[i][1], 1.0))
    
    return PauliSum(hamiltonian)


def hamiltonian_from_dict(data_dict):
    
    """
    Builds a Hamiltonian from a dict with keys indicating the pairs of connected vertices, 
    and values equal to the weights on the corresponding edges.
    """
    
    vertex_pairs = [*data_dict.keys()]
    edge_weights = [*data_dict.values()]
    
    hamiltonian = []

    for i in range(len(vertex_pairs)):
        
        qubit_a = int(vertex_pairs[i][0])
        qubit_b = int(vertex_pairs[i][1])
        hamiltonian.append(PauliTerm("Z",qubit_a,edge_weights[i])*PauliTerm("Z",qubit_b, 1.0))

    return PauliSum(hamiltonian)

"""

INCLUDE JL's other functions, eg create_normalized_random_hamiltonian?

Other possibilities:
    
    - Make the create_random_hamiltonian method more general by allowing the user to specify the sparsity of the graph (via some measure),
    or the ability to have full connectivity.
    
"""

## Methods for creating simple toy data sets

def distances_dataset(data):
    
    """
    Compute the pairwise Euclidean distance between data points in a specified dataset.
    The idea here is to take any dataset and get the weights to be used in (eg) a simple
    QAOA Maxcut.
    Could expand to include an arbitrary function of the Euclidean distance
    (eg with exponential decay).
    """
    
    if type(data) == dict:
        data = np.concatenate(list(data.values()))
    
    data = np.array(data)
    data_len = len(data)
    distances = {}
    for i in range(data_len):
        
        for j in range(i,data_len):
            
            if i==j:
                continue

            tmp_dict = {f"{i}{j}": np.linalg.norm(data[i] - data[j])}
            distances.update(tmp_dict)
            
    return distances

def create_gaussian_2Dclusters(n_clusters,n_points,means,variances,covs):
    
    """
    Description
    -----------
    Creates a set of clustered data points, where the distribution within each cluster is Gaussian.

    Parameters
    ----------
    :param      n_clusters:      The number of clusters
    :param      n_points:        A list of the number of points in each cluster
    :param      means:           A list of the means [x,y] coordinates of each cluster in the plane (i.e. their centre)
    :param      variances:       A list of the variances in the [x,y] coordinates of each cluster
    :param      covs:            A list of the covariances in the x and y coordinates of each cluster

    Returns
    -------
    :param      data             A dict whose keys are the cluster labels, and values are a matrix of the with the x and y coordinates as its rows.
    """
    args_in = [len(means),len(variances),len(covs),len(n_points)]
    assert all(item == n_clusters for item in args_in), "Insufficient data provided for specified number of clusters"
    
    data = {}
    for i in range(n_clusters):
        
        cluster_mean = means[i]
        cov_matr = [[variances[i][0], covs[i]],[covs[i],variances[i][1]]]
        
        x,y = np.random.multivariate_normal(cluster_mean,cov_matr,n_points[i]).T
        coords = np.array([x,y])
        tmp_dict = {str(i): coords.T}
        data.update(tmp_dict)
        
    return data

def plot_cluster_data(data):
    
    data_matr = np.concatenate(list(data.values()))
    plt.scatter(data_matr[:,0],data_matr[:,1])
    plt.show()

def create_circular_clusters(n_clusters,n_points,centres,radii):
    
    """
    Description
    -----------
    Creates a set of circularly clustered data points.
    In each cluster, points are uniformly sampled within each circle, with the
    specified centres and radii.

    Parameters
    ----------
    :param      n_clusters:      The number of clusters
    :param      n_points:        A list of the number of points in each cluster
    :param      centres:         [X,Y] coordinates of the centres of each circle
    :param      radii:           A list of the radii of the circles.

    Returns
    -------
    :param      data
    """
    
    args_in = [len(n_points),len(centres),len(radii)]
    assert all(item == n_clusters for item in args_in), "Insufficient data provided for specified number of clusters"
    
    a = np.linspace(0,2*np.pi,100)
    data = [], cluster_labels = []
    for i in range(n_clusters):
        
        points = n_points[i]
        r = radii[i]
        for j in range(points): 
              
            theta = np.random.choice(a)
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            data.append([x,y])
            cluster_labels += i
        
    data = np.array(data)
        
    return data, cluster_labels
    

## Methods for QAOA parameter landscape sweeps

def prepare_sweep_parameters(param1_2var,param1_range,param2_2var,param2_range,betas,gammas_singles,gammas_pairs):

    """
    TODO: WORK OUT WITH JL THE BEST WAY TO REQUEST PARAMETERS FROM USER
    """
    

    beta_var, beta_p = betas2var
    gamma_var, gamma_p = gammas2var

    params2var = len(beta_var) + len(gamma_var)

    assert all(i <= (n_qubits-1) for i in beta_var), "Cannot vary more beta coefficients than the number of qubits"
    assert all(i <= (len(coefficients)-1) for i in gamma_var), "Cannot vary more gamma coefficients than the number of Hamiltonian coupling terms"
    assert (params2var >= 1 and params2var <= 2), "Specified number of QAOA parameters to vary is zero or greater than two."
    assert all(i > 0 for i in beta_p + gamma_p), "QAOA step indices must be greater than zero"
    assert (max(list(beta_p) + list(gamma_p)) <= QAOA_p), "QAOA parameter specified to be varied in a step greater than the maximum number of steps."

    n_betas = int(len(betas)/QAOA_p)
    n_gammas = int(len(gammas)/QAOA_p)

    param_labels = []

    if beta_var:
        param_labels += ['Beta' + str(beta_var[i]) + '(' + str(beta_p[i]) + ')' for i in range(len(beta_var))]
        beta_p = [beta_p[i] - 1 for i in range(len(beta_p))]
        beta_var = [beta_var[i] + beta_p[i]*n_betas for i in range(len(beta_var))]
    else:
        beta_range2var = []

    if gamma_var:
        param_labels += ['Gamma' + str(gamma_var[i]) + '(' + str(gamma_p[i]) + ')' for i in range(len(gamma_var))]
        gamma_p = [gamma_p[i] - 1 for i in range(len(gamma_p))]
        gamma_var = [QAOA_p*n_betas + gamma_var[i] + gamma_p[i]*n_gammas for i in range(len(gamma_var))]
    else:
        gamma_range2var = []

    params = np.hstack((betas,gammas))

    if params2var == 1:
        param1 = beta_var + gamma_var
        param1_range = (list(beta_range2var) + list(gamma_range2var))[0]
        param2 = []
        param2_range = []
    else:
        param1,param2 = beta_var + gamma_var
        param1_range, param2_range = list(beta_range2var) + list(gamma_range2var)

    return params, param1, param2, param1_range, param2_range, param_labels
