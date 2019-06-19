import numpy as np
import networkx as nx

from pyquil.paulis import PauliSum, PauliTerm 
from math import log
from sklearn.metrics import accuracy_score

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
    """
    coins = np.random.randint(2, size=(nqubits + nqubits**2))
    coeffs = np.random.rand(nqubits + nqubits**2) * 2 - 1
    hamiltonian = PauliSum("0.0")
    if single_terms:
        for i in range(nqubits - 1):
            if coins[i]:
                hamiltonian += PauliTerm([["Z", i]], coeffs[i])
    hamiltonian += PauliTerm([["Z", nqubits - 1]], coeffs[nqubits - 1]
                             )  # make sure maxqubit is the right one

    if pair_terms:
        for i in range(nqubits):
            for j in range(i):
                hamiltonian += PauliTerm([["Z", j], ["Z", i]], coeffs[nqubits * i + j])
    return hamiltonian

"""
INCLUDE JL's other functions, eg create_normalized_random_hamiltonian?
"""

def distances_dataset(data):
    
    """
    Compute the pairwise Euclidean distance between data points in a specified dataset.
    The idea here is to take any dataset and get the weights to be used in (eg) a simple
    QAOA Maxcut.
    Could expand to include an arbitrary function of the Euclidean distance
    (eg with exponential decay).
    """
    
    data = np.array(data)
    data_len = len(data)
    distances = np.zeros()
    for i in range(data_len):
        
        for j in range(i,data_len):
        
            dist = np.linalg.norm(data[i] - data[j])
            distances.append(dist)  
            
    return distances

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
            G.add_edge(i,j,weight)
        
        i_pointer += vertices - i
        
    return G

def hamiltonian_from_edges(vertices,edge_weights):
    
    """
    Builds a Hamiltonian from a list of graph edge weights
    Input list should be of form: [edge weights from node1 to remaining (n-1) nodes, 
                                   edge weights from node2 to all remaining (n-2) nodes,
                                   etc...]
    """
    
    hamiltonian = PauliSum("0.0")
    i_pointer = 0
    for i in range(vertices):  
        
        for j in range(i,vertices):
            
            weight = edge_weights[i_pointer] + j
            hamiltonian += PauliTerm([["Z", i], ["Z", j]], weight)
            
        i_pointer += vertices - i
        
    return hamiltonian

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
    :param      variances:       A list of the standard deviations in the [x,y] coordinates of each cluster
    :param      covs:            A list of the covariances in the x and y coordinates of each cluster

    Returns
    -------
    :param      data
    """
    args_in = [len(means),len(variances),len(covs),len(n_points)]
    assert all(item == n_clusters for item in args_in), "Insufficient data provided for specified number of clusters"
    
    data = [], cluster_labels = []
    for i in range(n_clusters):
        
        cluster_mean = means[i]
        cov_matr = [[variances[i,0], covs[i]],[covs[i],variances[i,1]]]
        
        x,y = np.random.multivariate_normal(cluster_mean,cov_matr,n_points[i]).T
        data.append([x,y])
        cluster_labels += i
        
    data = np.array(data)
        
    return data, cluster_labels

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

def return_lowest_state(probs):
    '''
    Description
    -----------
    Returns the lowest energy state of a QAOA run from the list of probabilities
    returned by pyQuil's Wavefunction.probabilities()method.

    Parameters
    ----------
    :param      probs:      A numpy array of length 2^n, returned by Wavefunction.probabilities() 

    Returns
    -------
    :param      lowest:     A little endian list of binary integers indicating the lowest energy state of the wavefunction.
    '''
    index_max = max(range(len(probs)), key=probs.__getitem__)
    string = '{0:0'+str(int(log(len(probs),2)))+'b}'
    string = string.format(index_max)
    return [int(item) for item in string]

def evaluate_lowest_state(lowest, true):
    '''
    Description
    -----------
    Prints informative statements comparing QAOA's returned bit string to the true
    cluster values.

    Parameters
    ----------
    :param      lowest:      A littleiendian list of binary integers representing the lowest energy state of the wavefunction
    :param     true:        A little-endian list of binary integers representing the true solution to the MAXCUT clustering problem.

    Returns
    -------
    Nothing
    '''
    print('True Labels of samples:',true_clusters)
    print('Lowest QAOA State:',lowest)
    acc = accuracy_score(lowest,true_clusters)
    print('Accuracy of Original State:',acc*100,'%')
    final_c = [0 if item == 1 else 1 for item in lowest]
    acc_c = accuracy_score(final_c,true_clusters)
    print('Accuracy of Complement State:',acc_c*100,'%')

def generate_hamiltonian_from_dist(dist,biases=None):
    '''
    Description
    -----------
    Generates a hamiltonian from a distance matrix and a numpy array of single qubit bias terms where the i'th indexed value
    of in biases is applied to the i'th qubit. 

    Parameters
    ----------
    :param      dist:      A 2-dimensional square matrix where entries in row i, column j represent the distance between node i and node j.
    :param     biases:     A numpy array of length(dist), with non-zero entries indicating single-qubit bias terms.

    Returns
    -------
    :param     hamiltonian: A PauliSum object modelling the hamiltonian of the system  
    '''
    pauli_list = list()
    m,n = dist.shape

    #only if a list is passed in for biases
    if biases:
        if not isinstance(biases,type(list())) or isinstance(biases,np.ndarray):
           raise ValueError("biases must be of type list()")
        if not len(biases)==len(dist):
            raise ValueError("biases must be the same length as dist (one number for each qubit)")

        #single qubit interactions
        for i, num in enumerate(biases):
            term = PauliTerm("Z",i,num)
            pauli_list.append(term)

    #pairwise interactions
    for i in range(m):
        for j in range(n):
            if i < j:
                term = PauliTerm("Z",i,dist.values[i][j])*PauliTerm("Z",j, 1.0)
                pauli_list.append(term)
            
    
    
    return PauliSum(pauli_list)
