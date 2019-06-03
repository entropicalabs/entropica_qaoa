import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from pyquil.paulis import PauliSum, PauliTerm 
from qaoa.parameters import QAOAParameterIterator
from scipy.spatial import distance


### METHODS FOR CREATING RANDOM HAMILTONIANS AND GRAPHS, AND SWITCHING BETWEEN THE TWO ###

"""
General TODOs / considerations:
    
- Include JL's other functions, eg create_normalized_random_hamiltonian?
- Implement certain types of graphs (eg Farhi's ring of disagrees, Erdos-Renyi, scale-free, etc)

"""

def create_random_hamiltonian(nqubits):
    """
    Description
    -----------
    Creates a random hamiltonian.

    Parameters
    ----------
    :param      nqubits:            The number of qubits


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
    
    
    1) k-regular graphs (graphs where all nodes have the same degree)
    2) degree distribution (exponential, log-normal, power law)
    3) graph from some empirically observed degree distribution
    """
    hamiltonian = []

    numb_biases = np.random.randint(nqubits)
    bias_qubits = np.random.choice(nqubits,numb_biases,replace=False)
    bias_coeffs = np.random.rand(numb_biases)
    for i in range(numb_biases):
        hamiltonian.append(PauliTerm("Z", int(bias_qubits[i]), bias_coeffs[i]))


    for i in range(nqubits):
        for j in range(i+1,nqubits):
            are_coupled = np.random.randint(2)
            if are_coupled:
                couple_coeff = np.random.rand()    
                hamiltonian.append(PauliTerm("Z", i, couple_coeff)*PauliTerm("Z", j, 1.0))

    return PauliSum(hamiltonian)

def create_graph(vertices,edge_weights):
    
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

#def graph_from_hamiltonian(vertex_pairs,edge_weights):
#    
#    """
#    Creates a networkx graph on specified number of vertices, with the specified edge_weights
#    """
#    
#    G = nx.Graph()
#    
#    for i in range(len(vertex_pairs)):
#        G.add_edge(vertex_pairs[i][0],vertex_pairs[i][1],weight=edge_weights[i])
#        
#    return G

def graph_from_hamiltonian(hamiltonian):
    
    G = nx.Graph()
    dim = len(hamiltonian)
    for i in range(dim):
        qubits = hamiltonian.terms[i].get_qubits()
        if len(qubits) == 1:
            G.add_node(qubits[0], weight=hamiltonian.terms[i].coefficient)
        else:
            G.add_edge(qubits[0],qubits[1],weight=hamiltonian.terms[i].coefficient)
        
    return G

def plot_graph(G):
    
    """
    Takes in a networkx graph and plots it
    TODO: can we also take in the list of nodes with a bias, and somehow show this on the plot too?
    """
    
    weights = np.real([*nx.get_edge_attributes(G,'weight').values()])
    pos = nx.shell_layout(G)
    
    nx.draw(G,pos,node_color='#A0CBE2',with_labels=True,edge_color=weights,
                 width=4, edge_cmap=plt.cm.Blues)
    plt.show()

def hamiltonian_from_graph(G):
    
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

def hamiltonian_from_distance_matrix(matr):
    
    hamiltonian = []
    dim = len(matr)
    for i in range(dim):
        for j in range(i+1,dim):
            hamiltonian.append(PauliTerm("Z",i,matr[i][j])*PauliTerm("Z",j, 1.0))
      
    return PauliSum(hamiltonian)

### METHODS FOR CREATING SIMPLE TOY DATA SETS FOR MAXCUT CLUSTERING ###

def distances_dataset(data):
    
    """
    Compute the pairwise Euclidean distance between data points in a specified dataset.
    The idea here is to take any dataset and get the weights to be used in (eg) a simple
    QAOA Maxcut.
    Could expand to include an arbitrary function of the Euclidean distance
    (eg with exponential decay) - would just require passing in the desired distance metric for cdist.
    """
    
    if type(data) == dict:
        data = np.concatenate(list(data.values()))

    return distance.cdist(data, data, 'euclidean')

def create_gaussian_2Dclusters(n_clusters,n_points,means,cov_matrices):
    
    """
    Description
    -----------
    Creates a set of clustered data points, where the distribution within each cluster is Gaussian.

    Parameters
    ----------
    :param      n_clusters:      The number of clusters
    :param      n_points:        A list of the number of points in each cluster
    :param      means:           A list of the means [x,y] coordinates of each cluster in the plane (i.e. their centre)
    :param      cov_matrices:    A list of the covariance matrices of the clusters

    Returns
    -------
    :param      data             A dict whose keys are the cluster labels, and values are a matrix of the with the x and y coordinates as its rows.
    """
    args_in = [len(means),len(cov_matrices),len(n_points)]
    assert all(item == n_clusters for item in args_in), "Insufficient data provided for specified number of clusters"
    
    data = {}
    for i in range(n_clusters):
        
        cluster_mean = means[i]
        
        x,y = np.random.multivariate_normal(cluster_mean,cov_matrices[i],n_points[i]).T
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
    

### METHODS FOR QAOA PARAMETER LANDSCAPE SWEEPS ###

def prepare_sweep_parameters(param1_2var,param1_range,param2_2var,param2_range,betas,gammas_singles,gammas_pairs):

    """
    THIS STILL NEEDS TO BE FIXED (NOT USEABLE AT PRESENT)
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


### ANALYTIC FORMULAE ###
    
# etc 