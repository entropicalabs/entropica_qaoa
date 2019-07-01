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

def graph_from_hamiltonian(vertex_pairs,edge_weights):
    
    """
    Creates a networkx graph on specified number of vertices, with the specified edge_weights
    """
    
    G = nx.Graph()
    
    for i in range(len(vertex_pairs)):
        G.add_edge(vertex_pairs[i][0],vertex_pairs[i][1],weight=edge_weights[i])
        
    return G