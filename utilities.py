import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from pyquil import Program
from pyquil.paulis import PauliSum, PauliTerm 
from pyquil.gates import X
from scipy.spatial import distance



### METHODS FOR CREATING RANDOM HAMILTONIANS AND GRAPHS, AND SWITCHING BETWEEN THE TWO ###

"""
TODOs / considerations:
    
- Include JL's other functions, eg create_normalized_random_hamiltonian?
- Implement certain types of graphs (eg Erdos-Renyi, scale-free, etc)
- Improve distances_dataset so it can do more than just compute Euclidean distances.
- Keep create_circular_clusters method? If so, it likely needs some attention.
- Qubit placeholder use in Networkx graphs

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

def hamiltonian_from_distance_matrix(dist,biases=None):
    '''
    Description
    -----------
    Generates a hamiltonian from a distance matrix and a numpy array of single qubit bias terms where the i'th indexed value
    of in biases is applied to the i'th qubit. 

    Parameters
    ----------
    :param      dist:      A 2-dimensional square matrix where entries in row i, column j represent the distance between node i and node j.
    :param     biases:     A dictionary of floats, with keys indicating the qubits with bias terms, and corresponding values being the bias coefficients.

    Returns
    -------
    :param     hamiltonian: A PauliSum object modelling the hamiltonian of the system  
    '''
    pauli_list = list()
    m,n = dist.shape

    if biases:
        if not isinstance(biases,type(dict())):
           raise ValueError(“biases must be of type dict()“)        
        for key in biases:
            term = PauliTerm(“Z”,key,biases[key])
            pauli_list.append(term)

    #pairwise interactions
    for i in range(m):
        for j in range(n):
            if i < j:
                term = PauliTerm("Z",i,dist.values[i][j])*PauliTerm("Z",j, 1.0)
                pauli_list.append(term)
            
    return PauliSum(pauli_list)

def hamiltonian_from_hyperparams(nqubits,singles,biases,pairs,couplings):
    
    hamiltonian = []
    for i in range(len(pairs)):
        hamiltonian.append(PauliTerm("Z",pairs[i][0],couplings[i])*PauliTerm("Z",pairs[i][1]))  
    
    for i in range(len(singles)):
        hamiltonian.append(PauliTerm("Z",singles[i],biases[i]))  
        
    return PauliSum(hamiltonian)

def hamiltonian_from_hyperparams(nqubits,singles,biases,pairs,couplings):
    
    hamiltonian = []
    for i in range(len(pairs)):
        hamiltonian.append(PauliTerm("Z",pairs[i][0],couplings[i])*PauliTerm("Z",pairs[i][1]))  
    
    for i in range(len(singles)):
        hamiltonian.append(PauliTerm("Z",singles[i],biases[i]))  
        
    return PauliSum(hamiltonian)

def ring_of_disagrees(n):
    
    hamiltonian = []
    for i in range(n-1):
        hamiltonian.append(PauliTerm("Z",i,0.5)*PauliTerm("Z",i+1))  
    hamiltonian.append(PauliTerm("Z",n-1,0.5)*PauliTerm("Z", 0))
        
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
    
### ANALYTIC FORMULAE ###
    
# etc 

### OTHER MISCELLANEOUS ###
    
def prepare_classical_state(reg, state) -> Program:
    """Prepare a custom classical state for all qubits in reg.
     Parameters
    ----------
    state : Type[list]
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
    # if int(s) == 0 we don't need to add any gates, since the qubit is in state 0 by default
        if int(s) == 1:
            p.inst(X(qubit))
    return p
    
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
    :param      lowest:      A little-endian list of binary integers representing the lowest energy state of the wavefunction
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
    
def plot_amplitudes(amplitudes, energies, ax=None):
    """
    Description
    -----------
    Makes a nice plot of the probabilities for each state and its energy
    
    Parameters
    ----------
    :param amplitudes: (array/list) the probabilites to find the state
    :param energies:   (array/list) The energy of that state
    :ax:               (matplotlib axes object) The canvas to draw on
    """
    if ax == None:
        fig, ax = plt.subplots()
    # normalizing energies
    energies = np.array(energies)
    energies /= max(abs(energies))
    
    format_strings = ('{0:00b}', '{0:01b}', '{0:02b}', '{0:03b}', '{0:04b}', '{0:05b}')
    nqubits = int(np.log2(len(energies)))
    
    # create labels
    labels = [r'$\left|' + format_strings[nqubits].format(i) + r'\right>$' for i in range(len(amplitudes))]
    y_pos = np.arange(len(amplitudes))
    width = 0.35
    ax.bar(y_pos, amplitudes**2, width, label=r'$|Amplitude|^2$')
    
    ax.bar(y_pos+width, -energies, width, label="-Energy")
    ax.set_xticks(y_pos+width/2, minor=False)
    ax.set_xticklabels(labels, minor=False)
#    plt.ylabel("Amplitude")
    ax.set_xlabel("State")
    ax.grid(linestyle='--')
    ax.legend()
#    plt.show()
